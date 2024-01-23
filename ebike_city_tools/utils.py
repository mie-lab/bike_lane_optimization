import os
import networkx as nx
import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.spatial.distance import cdist
from shapely.geometry import LineString
from ebike_city_tools.graph_utils import (
    transfer_node_attributes,
    determine_vertices_on_shortest_paths,
    determine_arcs_between_vertices,
)


def output_to_dataframe(streetIP, G: nx.DiGraph, fixed_edges: pd.DataFrame = pd.DataFrame()) -> pd.DataFrame:
    """
    Convert the solution of the LP / IP into a dataframe with the optimal capacities
    Arguments:
        streetIP: mip.Model
        G: nx.DiGraph, street graph (same as used for the LP)
        fixed_edges: dataframe with fixed capacities that were not optimized
    Returns:
        capacities: pd.DataFrame with columns ("Edge", "u_b(e)", "u_c(e)", "capacity")
    """
    # Does not output a dataframe if mathematical program ⁄infeasible
    assert streetIP.objective_value is not None

    capacities = nx.get_edge_attributes(G, "capacity")

    # Creates the output dataframe
    # if fixed_values.empty:
    edge_cap = []

    def retrieve_u_for_fixed_edge(edge, column):
        row = fixed_edges.loc[fixed_edges["Edge"] == edge]
        u = row[column].values[0]
        return u

    for i, e in enumerate(G.edges):
        opt_cap_car = (
            streetIP.vars[f"u_{e},c"].x
            if not streetIP.vars[f"u_{e},c"] is None
            else retrieve_u_for_fixed_edge(e, "u_c(e)")
        )
        opt_cap_bike = (
            streetIP.vars[f"u_{e},b"].x
            if not streetIP.vars[f"u_{e},b"] is None
            else retrieve_u_for_fixed_edge(e, "u_b(e)")
        )
        edge_cap.append([e, opt_cap_bike, opt_cap_car, capacities[e]])
    dataframe_edge_cap = pd.DataFrame(data=edge_cap)
    dataframe_edge_cap.columns = ["Edge", "u_b(e)", "u_c(e)", "capacity"]
    return dataframe_edge_cap


def flow_to_df(streetIP):
    """
    Output all optimized flow variables in a dataframe
    Arguments:
        streetIP: mip.Model
    Returns:
        flow_df: pd.DataFrame with optimal flow for each (s, t, e, lanetype) combination
    """
    var_values = []
    for m in streetIP.vars:
        if m.name.startswith("f_"):
            (s, t, edge_u, edge_v, edgetype) = m.name[2:].split(",")
            edge_u = int(edge_u[1:])
            edge_v = int(edge_v.split(")")[0][1:])
            var_values.append([m.name, s, t, edge_u, edge_v, edgetype, m.x])
    flow_df = pd.DataFrame(var_values, columns=["name", "s", "t", "edge_u", "edge_v", "var_type", "flow"])
    return flow_df


def result_to_streets(capacity_df):
    """
    Convert a capacity dataframe with one row per lane into a dataframe with one row per street
    Arguments:
        capacity_df: pd.DataFrame with columns ("Edge", "u_b(e)", "u_c(e)", "capacity")
    Returns:
        street_capacity_df: pd.DataFrame with columns ("Edge", "u_b(e)", "u_c(e)", "u_c(e)_reversed", "capacity")
    """
    # if edge column is a string, we need to apply eval to convert it into a tuple
    if type(capacity_df.iloc[0]["Edge"]) == str:
        capacity_df["Edge"] = capacity_df["Edge"].apply(eval)
    capacity_df["source<target"] = capacity_df["Edge"].apply(lambda x: x[0] < x[1])
    capacity_df.set_index("Edge", inplace=True)
    # make new representation with all in one line
    reversed_edges = capacity_df[~capacity_df["source<target"]].reset_index()
    reversed_edges["Edge"] = reversed_edges["Edge"].apply(lambda x: (x[1], x[0]))
    together = pd.merge(
        capacity_df[capacity_df["source<target"]],
        reversed_edges,
        how="outer",
        left_index=True,
        right_on="Edge",
        suffixes=("", "_reversed"),
    ).set_index("Edge")
    assert all(together["capacity"] == together["capacity_reversed"])
    assert all(together["u_b(e)"] == together["u_b(e)_reversed"])
    assert all(together["source<target"] != together["source<target_reversed"])
    together.drop(
        ["capacity_reversed", "u_b(e)_reversed", "source<target_reversed", "source<target"], axis=1, inplace=True
    )
    assert len(together) == 0.5 * len(capacity_df)
    return together


def fix_multilane_bike_lanes(G_lane: nx.MultiDiGraph, check_for_existing: bool = False) -> list:
    """
    Takes a graph and converts one lane from all multi-lanes to a bike lane (if there is no bike lane yet)

    Args:
        G_lane (nx.MultiDiGraph): Input graph where some motorized lanes should be converted to bike lanes (inplace)
        check_for_existing (bool): whether to check if there is an existing bike lane

    Returns:
        List of edges (u,v,key) to be fixed as bike edges
    """
    streets_already_with_bike = set()
    edges_to_transform = []
    for u in G_lane.nodes():
        for v in G_lane.neighbors(u):
            # check whether there is a double lane AND there is also at least one edge in the other direction
            if G_lane.number_of_edges(u, v) > 1 and G_lane.number_of_edges(v, u) > 0:
                # if there is already a bike lane, skip (only if we don't reallocate existing bike lanes)
                if check_for_existing and "lanetype" in G_lane[u][v].items():
                    lanetypes = {k: d["lanetype"] for k, d in G_lane[u][v].items()}
                    if any(["P" in l for l, l in lanetypes.items()]):
                        continue
                # check if there is already a bidirectional bike lane placed on this street
                if (u, v) in streets_already_with_bike or (v, u) in streets_already_with_bike:
                    continue
                # iterate over the edges for this lane
                for key in list(dict(G_lane[u][v]).keys()):
                    # convert the first lane we see that is not fixed (if fixed lanes even exist)
                    if ("fixed" not in G_lane[u][v][key]) or (not G_lane[u][v][key]["fixed"]):
                        # G_lane[u][v][key]["lane"] = "P" # Transform directly?
                        # G_lane[u][v][key]["lanetype"] = "P"
                        edges_to_transform.append((u, v, key))
                        streets_already_with_bike.add((u, v))
                        # print("converted", u, v, key, "to bike lane")
                        break
    return edges_to_transform


def extend_od_circular(od, nodes):
    """
    Create new OD matrix that ensures connectivity by connecting one node to the next in a list
    od: pd.DataFrame, original OD with columns s, t and trips_per_day
    nodes: list, all nodes in the graph
    """
    # shuffle and convert to df
    new_od_paths = pd.DataFrame(nodes, columns=["s"]).sample(frac=1).reset_index(drop=True)
    # shift by one to ensure cirularity
    new_od_paths["t"] = new_od_paths["s"].shift(1)
    # fille nan
    new_od_paths.loc[0, "t"] = new_od_paths.iloc[-1]["s"]

    # concatenate and add flow of 0 to the new OD pairs
    od_new = pd.concat([od, new_od_paths]).fillna(0).astype(int)

    # make sure that no loops
    od_new = od_new[od_new["s"] != od_new["t"]]
    return od_new.drop_duplicates(subset=["s", "t"])


def extend_od_matrix(od, nodes):
    """
    Extend the OD matrix such that every node appears as s and every node appears as t
    od: initial OD matrix, represented as a pd.Dataframe
    nodes: list of nodes in the graph
    """

    def get_missing_nodes(od_df):
        nodes_not_in_s = [n for n in nodes if n not in od_df["s"].values]
        nodes_not_in_t = [n for n in nodes if n not in od_df["t"].values]
        return nodes_not_in_s, nodes_not_in_t

    # find missing nodes
    nodes_not_in_s, nodes_not_in_t = get_missing_nodes(od)
    min_len = min([len(nodes_not_in_s), len(nodes_not_in_t)])
    len_diff = max([len(nodes_not_in_s), len(nodes_not_in_t)]) - min_len
    # combine every node of the longer list with a random permutation of the smaller list and add up
    if min_len == len(nodes_not_in_t):
        shuffled_t = np.random.permutation(nodes_not_in_t)
        combined_nodes = np.concatenate(
            [
                np.stack([nodes_not_in_s[:min_len], shuffled_t]),
                np.stack([nodes_not_in_s[min_len:], shuffled_t[:len_diff]]),
            ],
            axis=1,
        )
    else:
        shuffled_s = np.random.permutation(nodes_not_in_s)
        combined_nodes = np.concatenate(
            [
                np.stack([shuffled_s, nodes_not_in_t[:min_len]]),
                np.stack([shuffled_s[:len_diff], nodes_not_in_t[min_len:]]),
            ],
            axis=1,
        )
    # transform to dataframe
    new_od_paths = pd.DataFrame(combined_nodes.swapaxes(1, 0), columns=["s", "t"])
    # concat and add a flow value of 0 since we don't want to optimize the travel time for these lanes
    od_new = pd.concat([od, new_od_paths]).fillna(0)

    # check again
    nodes_not_in_s, nodes_not_in_t = get_missing_nodes(od_new)
    assert len(nodes_not_in_s) == 0 and len(nodes_not_in_t) == 0
    return od_new


def compute_bike_time(distance, gradient):
    """
    Following the formula from Parkin and Rotheram (2010)
    """
    if gradient > 0:
        # gradient positive -> reduced speed
        return distance / max([21.6 - 1.44 * gradient, 1])  # at least 1kmh speed
    else:
        # gradient positive -> increase speed (but - because gradient also negative)
        return distance / (21.6 - 0.86 * gradient)


def compute_car_time(row):
    if "M" in row["lanetype"]:
        return 60 * row["distance"] / row["speed_limit"]
    else:
        return np.inf


def compute_edgedependent_bike_time(row, shared_lane_factor: int = 2):
    """Following the formula from Parkin and Rotheram (2010)"""
    biketime = 60 * compute_bike_time(row["distance"], row["gradient"])
    if "P" in row["lanetype"]:
        return biketime
    else:
        return biketime * shared_lane_factor


def compute_penalized_car_time(attr_dict: dict, bike_lane_speed: int = 10) -> int:
    if "M" in attr_dict["lanetype"]:
        return 60 * attr_dict["distance"] / attr_dict["speed_limit"]
    elif "P" in attr_dict["lanetype"]:
        # return np.inf  # uncomment to test top-down approach with infinite paths
        return 60 * attr_dict["distance"] / bike_lane_speed  # assume speed limit on bike priority lanes
    else:
        raise RuntimeError("lanetyp other than M and P not implemented")


def output_lane_graph(
    G_lane,
    bike_G,
    car_G,
    shared_lane_factor=2,
    output_attr=["width", "distance", "length", "speed_limit", "fixed", "gradient"],
):
    """
    Output a lane graph in the format of the SNMan standard.
    Arguments:
        G_lane: Input lane graph (original street network) -> this is used to get the edge attributes
        car_G: nx.MultiDiGraph, Output graph of car network
        bike_G: nx.MultiGraph, Output graph of bike network
    """

    assert bike_G.number_of_edges() + car_G.number_of_edges() == G_lane.number_of_edges()

    # Step 1: make one dataframe of all car and bike edges
    car_edges = nx.to_pandas_edgelist(car_G)
    car_edges["lane"] = "M>"
    car_edges["lanetype"] = "M>"
    bike_edges = nx.to_pandas_edgelist(bike_G.to_directed())
    bike_edges["lane"] = "P>"
    bike_edges["lanetype"] = "P>"
    all_edges = pd.concat([car_edges, bike_edges])
    all_edges["direction"] = ">"
    all_edges.drop(output_attr, axis=1, inplace=True, errors="ignore")

    # Step 2: get all attributes of the original edges (in both directions)
    edges_G_lane = nx.to_pandas_edgelist(G_lane)
    # revert edges
    edges_G_lane_reversed = edges_G_lane.copy()
    edges_G_lane_reversed["gradient"] *= -1
    edges_G_lane_reversed["source_temp"] = edges_G_lane_reversed["source"]
    edges_G_lane_reversed["source"] = edges_G_lane_reversed["target"]
    edges_G_lane_reversed["target"] = edges_G_lane_reversed["source_temp"]
    # concat
    edges_G_lane_doubled = pd.concat([edges_G_lane, edges_G_lane_reversed])
    # extract relevant attributes --> these are all attributes that we can merge with the others
    agg_dict = {attr: "first" for attr in output_attr if attr in edges_G_lane_doubled.columns}
    edges_G_lane_doubled = edges_G_lane_doubled.groupby(["source", "target"]).agg(agg_dict)

    # Step 3: Merge with the attributes
    all_edges_with_attributes = all_edges.merge(
        edges_G_lane_doubled, how="left", left_on=["source", "target"], right_on=["source", "target"]
    )
    # Step 4: compute bike and car time
    all_edges_with_attributes["car_time"] = all_edges_with_attributes.apply(compute_car_time, axis=1)
    all_edges_with_attributes["bike_time"] = all_edges_with_attributes.apply(
        compute_edgedependent_bike_time, shared_lane_factor=shared_lane_factor, axis=1
    )

    # Step 5: make a graph
    attrs = [c for c in all_edges_with_attributes.columns if c not in ["source", "target"]]
    G_final = nx.from_pandas_edgelist(
        all_edges_with_attributes, edge_attr=attrs, source="source", target="target", create_using=nx.MultiDiGraph
    )
    # make sure that the other attributes are transferred to the new graph
    G_final = transfer_node_attributes(G_lane, G_final)
    return G_final


def match_od_with_nodes(station_data_path: str, nodes: gpd.GeoDataFrame):
    """
    Match a OD matrix of coordinates with the node IDs

    Args:
        station_data_path (str): path to folder for one city (district)
        nodes (gpd.GeoDataFrame): List of graph nodes with geometry
    Returns:
        pd.DataFrame with columns s, t and trips, containint origin and destination node id and the number of trips
    """

    def linestring_from_coords(row):
        return LineString([[row["start_lng"], row["start_lat"]], [row["end_lng"], row["end_lat"]]])

    station_data = pd.read_csv(station_data_path)
    print("Whole city OD matrix", len(station_data))

    # create linestring and convert to geodataframe
    station_data["geometry"] = station_data.apply(linestring_from_coords, axis=1)
    station_data = gpd.GeoDataFrame(station_data)
    station_data.set_geometry("geometry", inplace=True)

    if "birchplatz" in station_data_path or "affoltern" in station_data_path:
        original_crs = "EPSG:2056"
        station_data.crs = original_crs
        nodes.to_crs(2056, inplace=True)
    else:
        original_crs = "EPSG:4326"
        station_data.crs = original_crs
        if "cambridge" in station_data_path:
            nodes.to_crs("EPSG:2249", inplace=True)
            station_data.to_crs("EPSG:2249", inplace=True)
        elif "chicago" in station_data_path:
            nodes.to_crs("EPSG:26971", inplace=True)
            station_data.to_crs("EPSG:26971", inplace=True)
        else:
            raise NotImplementedError("Unknown city")
        station_data = station_data[station_data.geometry.is_valid]

    # select only the rows where the linestring intersects the area polygon
    area_polygon = gpd.GeoDataFrame(geometry=[nodes.geometry.unary_union.convex_hull], crs=nodes.crs)
    trips = station_data.sjoin(area_polygon)

    print("Number of trips intersetcing the area", len(trips))

    # get the closest nodes to the respective destination
    trips["geom_destination"] = gpd.points_from_xy(x=trips["end_lng"], y=trips["end_lat"])
    trips.set_geometry("geom_destination", inplace=True, crs=original_crs)
    trips.to_crs(nodes.crs, inplace=True)
    trips = trips.sjoin_nearest(nodes, distance_col="dist_destination", how="left", lsuffix="", rsuffix="destination")
    trips.rename(columns={"osmid": "osmid_destination"}, inplace=True)

    # set geometry to origin
    trips["geom_origin"] = gpd.points_from_xy(x=trips["start_lng"], y=trips["start_lat"])
    # trips["geom_origin"].apply(wkt.loads)
    trips.set_geometry("geom_origin", inplace=True, crs=original_crs)
    trips.to_crs(nodes.crs, inplace=True)
    trips.drop(["geom_destination"], axis=1, inplace=True)

    # get the closest nodes to the respective origin
    trips = trips.sjoin_nearest(nodes, distance_col="dist_origin", how="left", lsuffix="", rsuffix="origin")
    trips.rename(columns={"osmid": "osmid_origin"}, inplace=True)
    trips_final = (
        trips.groupby(["osmid_origin", "osmid_destination"])
        .agg({"count": "sum"})
        .reset_index()
        .rename(columns={"osmid_origin": "s", "osmid_destination": "t", "count": "trips"})
    )
    print(len(trips_final), trips_final["trips"].sum())
    return trips_final


def determine_valid_arcs(od_pairs, graph, number_of_paths=3):
    """Given a graph, a pair of vertices (s,t) and a specified number of paths to be specified, this returns
    the arcs that lie on one of the number_of_paths shortest s-t-paths."""

    output = {}
    od_pairs_as_pairs = list(zip(od_pairs["s"], od_pairs["t"]))
    for od_pair in od_pairs_as_pairs:
        vertices = determine_vertices_on_shortest_paths(od_pair, graph, number_of_paths)
        arcs = determine_arcs_between_vertices(graph, vertices)
        output[od_pair] = arcs
    return output


def make_node_ranking(G_base: nx.DiGraph):
    """
    Auxiliary method to sort nodes by their distance to other nodes

    Args:
        G_base (nx.DiGraph): Input graph

    Returns:
        distance_ranking (np.ndarray): 2D array with ranks of other nodes per node
        id_index_mapping (dict): maps from node ID to index in distance_ranking array
    """
    # get node attribute and save nodes in array
    node_coords = nx.get_node_attributes(G_base, "loc")
    id_index_mapping, index_id_mapping, node_coord_array = (
        {},
        np.zeros(len(node_coords)),
        np.zeros((len(node_coords), 2)),
    )
    for i, (key, value) in enumerate(sorted(node_coords.items())):
        id_index_mapping[key] = i  # map node ID to index in new array
        node_coord_array[i] = value[:2]
        index_id_mapping[i] = key

    # compute pairwise distance
    pairwise_distance = cdist(node_coord_array, node_coord_array)
    # find closest nodes
    distance_ranking = np.argsort(pairwise_distance, axis=1)
    # translate into closest node ID
    for i in range(len(distance_ranking)):
        distance_ranking[i] = index_id_mapping[(distance_ranking[i]).astype(int)]
    return distance_ranking.astype(int), id_index_mapping


def valid_arcs_spatial_selection(od: pd.DataFrame, G_base: nx.DiGraph, k_closest: int):
    """
    Select arcs for each OD pair by using the k_closest nodes for each node on the shortest path

    Args:
        od (pd.DataFrame): OD matrix
        G_base (nx.DiGraph): input graph
        k_closest (int): number of closest nodes selected for each node on the path

    Returns:
        dict: _description_
    """
    assert k_closest > 0, "k closest must be at least 0"

    # rank nodes by their distance to another
    distance_ranking, id_index_mapping = make_node_ranking(G_base)

    valid_arcs = {}
    # iterate over OD pairs
    for od_pair in zip(od["s"], od["t"]):
        (s, t) = od_pair
        # compute shortest path between them
        shortest_path_nodes = nx.shortest_path(G_base, s, t, "bike_travel_time")
        # for each node from the shortest path, find the k closest nodes
        nodes_for_subgraph = []
        for node in shortest_path_nodes:
            k_closest_nodes = distance_ranking[id_index_mapping[node], :k_closest]
            nodes_for_subgraph.extend(k_closest_nodes)
        # make the subgraph of all selected nodes
        valid_arcs[od_pair] = determine_arcs_between_vertices(G_base, nodes_for_subgraph)
    return valid_arcs
