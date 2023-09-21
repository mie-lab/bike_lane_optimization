import time
import os
import argparse
import pandas as pd
from ebike_city_tools.optimize.utils import make_fake_od, output_to_dataframe, flow_to_df
from ebike_city_tools.optimize.linear_program import define_IP
from ebike_city_tools.utils import lane_to_street_graph, extend_od_circular, output_lane_graph, filter_by_attribute
from ebike_city_tools.optimize.round_simple import pareto_frontier, rounding_and_splitting
from ebike_city_tools.iterative_algorithms import betweenness_pareto
import numpy as np
import geopandas as gpd
import networkx as nx
from snman import distribution, street_graph, graph_utils, io, merge_edges
from snman.constants import *

FLOW_CONSTANT = 1  # how much flow to send through a path


def deprecated_table_to_graph(
    edge_table, node_table=None, edge_attributes={"width_total_m": "capacity"}, node_attributes={"geometry": "location"}
):
    """
    DEPRECATED
    edge_table: pd.DataFrame with columns u, v, and the required edge attributes
    node_table (Optional): table with the node id as the index column and the edge id as the
    edge_attributes: Dictionary of the form {columns-name-in-table : desired-attribute_name-in-graph}
    node_attributes: Dictionary of the form {columns-name-in-table : desired-attribute_name-in-graph}
    """
    # init graph
    G = nx.DiGraph()

    # add edge list
    edge_list = []
    for row_ind, edge_row in edge_table.iterrows():
        # extract the edge attributes
        edge_attr = {attr_name: edge_row[col_name] for col_name, attr_name in edge_attributes.items()}
        # add edge with attributes to the list
        edge_list.append([edge_row["u"], edge_row["v"], edge_attr])
        edge_attr["gradient"] = -edge_attr["gradient"]
        edge_list.append([edge_row["v"], edge_row["u"], edge_attr])
    G.add_edges_from(edge_list)

    # set node attributes
    node_attrs = {}
    for row_ind, node_row in node_table.iterrows():
        node_attrs[row_ind] = {attr_name: node_row[col_name] for col_name, attr_name in node_attributes.items()}
    nx.set_node_attributes(G, node_attrs)
    return G


def deprecated_load_graph(path):
    # load nodes and edges
    nodes = gpd.read_file(os.path.join(path, "nodes_all_attributes.gpkg")).set_index("osmid")
    edges = gpd.read_file(os.path.join(path, "edges_all_attributes.gpkg"))
    edges = edges[["u", "v", "width_total_m", "maxspeed", "lanes", "length"]]
    # remove the ones with start and end at the same point
    edges = edges[edges["u"] != edges["v"]]
    # there are many 1.8 and 0.9 wide streets -> transform into 1 and 2 lane streets
    edges["width_total_m"] = edges["width_total_m"].round()  # TODO
    # fill nans of the capacity with 1
    # edges["lanes"] = edges["lanes"].fillna(1)

    # compute gradient
    gradient = []
    for i in range(len(edges)):
        gradient.append(
            100 * (nodes["elevation"][edges.iloc[i, 1]] - nodes["elevation"][edges.iloc[i, 0]]) / edges.iloc[i, 5]
        )
    edges["gradient"] = gradient

    # construct graph
    G = deprecated_table_to_graph(
        edges, nodes, {"width_total_m": "capacity", "length": "distance", "gradient": "gradient"}
    )
    return G


def generate_motorized_lane_graph(
    edge_path,
    node_path,
    source_lanes_attribute=KEY_LANES_DESCRIPTION,
    target_lanes_attribute=KEY_LANES_DESCRIPTION_AFTER,
):
    G = io.load_street_graph(edge_path, node_path)  # initialize lanes after rebuild
    # need to save the maxspeed attribute here to use it later
    maxspeed = nx.get_edge_attributes(G, "maxspeed")
    nx.set_edge_attributes(G, nx.get_edge_attributes(G, source_lanes_attribute), target_lanes_attribute)
    # ensure consistent edge directions
    street_graph.organize_edge_directions(G)

    distribution.set_given_lanes(G)
    H = street_graph.filter_lanes_by_modes(G, {MODE_PRIVATE_CARS}, lane_description_key=KEY_GIVEN_LANES_DESCRIPTION)

    merge_edges.reset_intermediate_nodes(H)
    merge_edges.merge_consecutive_edges(H, distinction_attributes={KEY_LANES_DESCRIPTION_AFTER})
    # make lane graph
    L = street_graph.to_lane_graph(H, KEY_GIVEN_LANES_DESCRIPTION)
    # make sure that the graph is strongly connected
    L = graph_utils.keep_only_the_largest_connected_component(L)
    # add some edge attributes that we need for the optimization (e.g. slope)
    L = adapt_edge_attributes(L, maxspeed)
    return L


def adapt_edge_attributes(L: nx.MultiDiGraph, maxspeed: dict):
    """Sets new edge attributes to be used in the optimization algorithm"""
    # add capacity attribute to 1 for all lanes
    nx.set_edge_attributes(L, 1, name="capacity")

    # rename to distance
    distances = nx.get_edge_attributes(L, "length")
    distances = {k: v / 1000 for k, v in distances.items()}  # transform to km
    nx.set_edge_attributes(L, distances, "distance")

    # add gradient attribute
    elevation_dict = nx.get_node_attributes(L, "elevation")
    length_dict = nx.get_edge_attributes(L, "length")
    gradient_attr, speed_limit_attr = {}, {}
    isna = 0
    for e in L.edges:
        (u, v, _) = e
        gradient_attr[e] = 100 * (elevation_dict[v] - elevation_dict[u]) / length_dict[e]
        # set speed limit attribute
        if (u, v, 0) in maxspeed.keys():
            speed_limit_attr[e] = maxspeed[(u, v, 0)]
        elif (v, u, 0) in maxspeed.keys():
            speed_limit_attr[e] = maxspeed[(v, u, 0)]
        else:
            speed_limit_attr[e] = pd.NA
        if pd.isna(speed_limit_attr[e]):
            isna += 1
            speed_limit_attr[e] = 30
    nx.set_edge_attributes(L, speed_limit_attr, name="speed_limit")
    nx.set_edge_attributes(L, gradient_attr, name="gradient")

    return L


def lane_optimization(L, maxspeed, od_df=None, edge_fraction=0.4, shared_lane_factor=2):
    """
    Optimizes bike lane allocation with LP approach - to be incorporated in SNMan
    Inputs:
        L: nx.MultiDiGraph, input graph with one edge per lane
        od_df: pd.DataFrame, dataframe with OD pairs (must have columns named 's' and 't')
        edge_fraction: Fraction of edges that should be eliminated (= converted to bike lanes)
    Returns:
        L_optimized: nx.MultiDiGraph, graph where certain car lanes were deleted (the ones that will be bike lanes)
        and where the car lane directions are optimized
    """
    # add some edge attributes that we need for the optimization (e.g. slope)
    G_lane = adapt_edge_attributes(L, maxspeed)
    if od_df is not None:
        od = od_df.copy()
    else:
        # Initialize od with empty dataframe
        od = pd.DataFrame(columns=["s", "t", "trips_per_day"])
    # reduce OD matrix to nodes that are in G_lane
    node_list = list(G_lane.nodes())
    od = od[(od["s"].isin(node_list)) & (od["t"].isin(node_list))]
    # extend OD matrix because otherwise we get disconnected car graph
    od = extend_od_circular(od, node_list)

    # change representation
    G_street = lane_to_street_graph(G_lane)

    ip = define_IP(
        G_street,
        cap_factor=1,
        car_weight=0.0001,
        od_df=od,
        bike_flow_constant=FLOW_CONSTANT,
        car_flow_constant=FLOW_CONSTANT,
        shared_lane_factor=shared_lane_factor,
        weight_od_flow=WEIGHT_OD_FLOW,
    )
    ip.optimize()
    capacity_values = output_to_dataframe(ip, G_street)
    # print("HERE", sum(capacity_values["u_b(e)"] > 0))

    # compute how many bike edges we want to have
    bike_edges_to_add = int(edge_fraction * G_lane.number_of_edges())
    capacity_values.to_csv("outputs/cap_lukas.csv")
    nx.write_gpickle(G_lane, "outputs/G_lukas.gpickle")

    # rounding algorithm
    bike_G, car_G = rounding_and_splitting(capacity_values, bike_edges_to_add=bike_edges_to_add)

    # combine bike and car graph in a new lane graph with all relevant attributes
    new_lane_graph = output_lane_graph(G_lane, bike_G, car_G, shared_lane_factor)

    # return only the car lanes
    G_filtered = filter_by_attribute(new_lane_graph, "lane", "M>")

    return G_filtered


if __name__ == "__main__":
    np.random.seed(42)
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data_path", default="../street_network_data/zollikerberg", type=str)
    parser.add_argument("-o", "--out_path", default="outputs", type=str)
    parser.add_argument("-b", "--run_betweenness", action="store_true")
    parser.add_argument(
        "-p", "--penalty_shared", default=2, type=int, help="penalty factor for driving on a car lane by bike"
    )
    parser.add_argument(
        "-s", "--sp_method", default="od", type=str, help="Compute the shortest path either 'all_pairs' or 'od'"
    )
    args = parser.parse_args()

    path = args.data_path
    shared_lane_factor = args.penalty_shared  # how much to penalize biking on car lanes
    OUT_PATH = args.out_path
    SP_METHOD = args.sp_method
    out_path_ending = "_od" if SP_METHOD == "od" else ""
    WEIGHT_OD_FLOW = False
    os.makedirs(OUT_PATH, exist_ok=True)

    # generate lane graph with snman
    G_lane = generate_motorized_lane_graph(
        os.path.join(path, "edges_all_attributes.gpkg"), os.path.join(path, "nodes_all_attributes.gpkg")
    )

    # load OD
    od = pd.read_csv(os.path.join(path, "od_matrix.csv"))
    od.rename({"osmid_origin": "s", "osmid_destination": "t"}, inplace=True, axis=1)
    od = od[od["s"] != od["t"]]
    # reduce OD matrix to nodes that are in G_lane
    node_list = list(G_lane.nodes())
    od = od[(od["s"].isin(node_list)) & (od["t"].isin(node_list))]

    # extend OD matrix because otherwise we get disconnected car graph
    od = extend_od_circular(od, node_list)

    # # making a subgraph only disconnects the graoh
    # nodes = nodes.sample(200)
    # edges = edges[edges["u"].isin(nodes.index)]
    # edges = edges[edges["v"].isin(nodes.index)]
    # od = od[od["s"].isin(nodes.index)]
    # od = od[od["t"].isin(nodes.index)]

    assert nx.is_strongly_connected(G_lane), "G not connected"

    if args.run_betweenness:
        print("Running betweenness algorithm for pareto frontier...")
        # run betweenness centrality algorithm for comparison
        pareto_between = betweenness_pareto(G_lane, sp_method=SP_METHOD, od_matrix=od, weight_od_flow=WEIGHT_OD_FLOW)
        pareto_between.to_csv(os.path.join(OUT_PATH, f"real_pareto_betweenness{out_path_ending}.csv"), index=False)

    G_street = lane_to_street_graph(G_lane)

    print("Running LP for pareto frontier...")
    tic = time.time()
    ip = define_IP(
        G_street,
        cap_factor=1,
        od_df=od,
        bike_flow_constant=FLOW_CONSTANT,
        car_flow_constant=FLOW_CONSTANT,
        shared_lane_factor=shared_lane_factor,
        weight_od_flow=WEIGHT_OD_FLOW,
    )
    toc = time.time()
    print("Finish init", toc - tic)
    ip.optimize()
    toc2 = time.time()
    print("Finish optimization", toc2 - toc)
    print("OPT VALUE", ip.objective_value)

    # nx.write_gpickle(G, "outputs/real_G.gpickle")
    capacity_values = output_to_dataframe(ip, G_street)
    capacity_values.to_csv(os.path.join(OUT_PATH, "real_u_solution.csv"), index=False)
    del ip
    # flow_df = flow_to_df(ip, list(G_street.edges))
    # flow_df.to_csv(os.path.join(OUT_PATH, "real_flow_solution.csv"), index=False)

    # compute the paretor frontier
    tic = time.time()
    pareto_df = pareto_frontier(
        G_lane,
        capacity_values,
        shared_lane_factor=shared_lane_factor,
        sp_method=SP_METHOD,
        od_matrix=od,
        weight_od_flow=WEIGHT_OD_FLOW,
    )
    print("Time pareto", time.time() - tic)
    pareto_df.to_csv(os.path.join(OUT_PATH, f"real_pareto_df{out_path_ending}.csv"), index=False)
