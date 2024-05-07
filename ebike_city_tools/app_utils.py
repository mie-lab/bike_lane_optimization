import os
import json
import numpy as np
import pandas as pd
import geopandas as gpd
import networkx as nx

from collections import Counter
from osmnx.bearing import add_edge_bearings, calculate_bearing
from sqlalchemy import create_engine
import psycopg2
from ebike_city_tools.utils import compute_edgedependent_bike_time, compute_car_time

CRS = 2056


# Setup database access
def get_database_connector(dblogin_file: str):
    """Create database connection with login json file"""
    with open(dblogin_file, "r") as infile:
        db_login = json.load(infile)
        db_login["database"] = "ebikecity"

    def get_con_mie():
        return psycopg2.connect(**db_login)

    return create_engine("postgresql+psycopg2://", creator=get_con_mie)


def get_expected_time_linear(nr_variables: int, coef=2.17235844e-05, intercept=-15.15954725242839):
    """Fit linear function to runtime from number of variables"""
    return nr_variables * coef + intercept


def get_expected_time(nr_variables: int):
    """Computes expected runtime (in min) from number of variables"""
    return 0.8 * np.exp(nr_variables / 1000000 * 1.6)


def compute_nr_variables(nr_edges: int, od_len: int):
    """Compute number of variables"""
    nr_variables = 3 * (od_len * nr_edges) + 2 * nr_edges
    return nr_variables


def generate_od_nodes(od_whole_zurich_nodes: pd.DataFrame, nodes: gpd.GeoDataFrame):
    """
    Fast method for generating the OD matrix for a specific area: Take the OD matrix for the whole city and only use
    the node-pairs of nodes that appear in the nodes-dataframe
    """
    assert nodes.index.name == "osmid"
    nodes_in_area = nodes.index.unique()
    od_in_area = od_whole_zurich_nodes[
        (od_whole_zurich_nodes["s"].isin(nodes_in_area)) & (od_whole_zurich_nodes["t"].isin(nodes_in_area))
    ]
    return od_in_area


def generate_od_geometry(area_polygon: gpd.GeoDataFrame, trips_microcensus: gpd.GeoDataFrame, nodes: gpd.GeoDataFrame):
    """
    Slow method for generating the OD matrix for a specific area: Take the Mobility Microncensus data, and match it
    to the nodes in this area.
    """
    # restrict trips to the ones crossing the area
    trips = trips_microcensus.sjoin(area_polygon)

    # get the closest nodes to the respective destination
    trips.set_geometry("geom_destination", inplace=True, crs=CRS)
    trips.to_crs(nodes.crs, inplace=True)
    trips = trips.sjoin_nearest(nodes, distance_col="dist_destination", how="left", lsuffix="", rsuffix="destination")
    trips.rename(columns={"osmid": "osmid_destination"}, inplace=True)

    # trips["geom_origin"].apply(wkt.loads)
    trips.set_geometry("geom_origin", inplace=True, crs=CRS)
    trips.to_crs(nodes.crs, inplace=True)
    trips.drop(["geom_destination"], axis=1, inplace=True)

    # get the closest nodes to the respective origin
    trips = trips.sjoin_nearest(nodes, distance_col="dist_origin", how="left", lsuffix="", rsuffix="origin")
    trips.rename(columns={"osmid": "osmid_origin"}, inplace=True)
    od_in_area = (
        trips.groupby(["osmid_origin", "osmid_destination"])
        .agg({"count": "sum"})
        .reset_index()
        .rename(columns={"osmid_origin": "s", "osmid_destination": "t", "count": "trips"})
    )
    return od_in_area


def recreate_lane_graph(project_edges: pd.DataFrame, run_output: pd.DataFrame):
    """Auxiliary method to create the graph from the project edges and the output of one run"""
    # set index
    project_edges["edge_key"] = project_edges["edge_key"].astype(str)
    if "source" in project_edges or "s" in project_edges:
        project_edges.set_index(["source", "target", "edge_key"], inplace=True)
    project_edges.sort_index(inplace=True)

    # add additional edges for new lanes (reversed bike lanes in the other direction)
    reversed_bike_edges = run_output[(run_output["lanetype"] == "P") & (run_output["edge_key"].str.contains("revbike"))]
    new_edges = []
    for _, bike_reversed_row in reversed_bike_edges.iterrows():
        # print(bike_reversed_row)
        s, t, k = bike_reversed_row["source"], bike_reversed_row["target"], bike_reversed_row["edge_key"]
        forward_row = project_edges.loc[t, s].iloc[0].to_dict()
        forward_row["gradient"] = forward_row["gradient"] * (-1)
        forward_row["source"] = s
        forward_row["target"] = t
        forward_row["lanetype"] = "P"
        forward_row["edge_key"] = k
        new_edges.append(forward_row)

    replaced_bike_edges = run_output[
        (run_output["lanetype"] == "P") & ~(run_output["edge_key"].str.contains("revbike"))
    ]

    counter = 0
    for _, bike_forward_row in replaced_bike_edges.iterrows():
        s, t, k = bike_forward_row["source"], bike_forward_row["target"], str(bike_forward_row["edge_key"])
        project_edges.loc[(s, t, k), "lanetype"] = "P"
        counter += 1
    # print(f"converted {counter} edges")
    total_edges = pd.concat((project_edges.reset_index(), pd.DataFrame(new_edges)))
    # print(len(total_edges), len(project_edges), sum(total_edges["lanetype"] == "P"))

    # add bike and car time attributes
    total_edges["bike_time"] = total_edges.apply(compute_edgedependent_bike_time, axis=1)
    total_edges["car_time"] = total_edges.apply(compute_car_time, axis=1)

    lane_graph = nx.from_pandas_edgelist(
        total_edges,
        edge_key="edge_key",
        edge_attr=[col for col in total_edges.columns if col not in ["source", "target", "edge_key"]],
        create_using=nx.MultiDiGraph,
    )
    return lane_graph

### complexity ###
def get_mode_subgraph(lane_graph, mode):
    """
    Constructs a subgraph for one transport mode.
    """
    G_edges = [(s, t, k) for s, t, k, data in lane_graph.edges(keys=True, data=True) if mode in data.get('lanetype', '')]
    G = lane_graph.edge_subgraph(G_edges)
    
    return G 

def get_degree_ratios(lane_graph, mode):
    """
    Calculates different node degree ratios as a dictionery with degree as key and ratio as value.
    A drawback is that ratios exlude degrees from the other mode dubgraph.
    """
    G = get_mode_subgraph(lane_graph, mode)
    
    d_values = dict(G.degree())
    d_counts = Counter(d_values.values())
    d_ratios = {degree: count / len(G.nodes()) for degree, count in d_counts.items()}
    
    return dict(sorted(d_ratios.items()))


## bearing ###
# bearing of a direct vector between OD ('as crow flies' line).
def calculate_path_bearing(node1, node2):
    return calculate_bearing(node1['x'], node1['y'], node2['x'], node2['y'])

# individual edge deviations for path.
def average_path_deviation(G, path):
    path_bearing = calculate_path_bearing(G.nodes[path[0]], G.nodes[path[-1]])
    deviations = [angle_difference(path_bearing, list(G.get_edge_data(path[i], path[i+1]).values())[0]['bearing'])
                  for i in range(len(path) - 1)]
    return np.mean(deviations)

# When calculating the bearing difference, 
# ensure that the result is always the smallest angle between the two bearings, 
# which should be between 0 and 180 degrees.

def angle_difference(bearing1, bearing2):
    diff = abs(bearing1 - bearing2) % 360
    return min(diff, 360 - diff)


def get_network_bearings(lane_graph, mode, weight=None):
    """
    Calculates and returns the average deviation of all shortest path edges from the direct bearing
    between nodes in a specified mode subgraph of a transportation network.
    """
    G = get_mode_subgraph(lane_graph, mode)

    G = add_edge_bearings(G)  

    all_pairs_sp = nx.all_pairs_dijkstra_path(G, weight=weight)
    total_path_deviation = 0
    path_count = 0
                
    for start, paths in all_pairs_sp:
        for end, path in paths.items():
            if start == end and len(path) <= 2:
                continue
            
            average_deviation = average_path_deviation(G, path)
            total_path_deviation += average_deviation
            path_count += 1            

    return total_path_deviation / path_count
