import networkx as nx
import pandas as pd
import numpy as np
import os
import warnings

from ebike_city_tools.utils import (
    extend_od_circular,
    match_od_with_nodes,
)
from ebike_city_tools.graph_utils import (
    filter_by_attribute,
    remove_node_attribute,
    nodes_to_geodataframe,
)
from ebike_city_tools.optimize.round_optimized import ParetoRoundOptimize
from snman import distribution, street_graph, graph, io, merge_edges, lane_graph, rebuilding
from snman.constants import (
    KEY_LANES_DESCRIPTION,
    KEY_LANES_DESCRIPTION_AFTER,
    MODE_PRIVATE_CARS,
    KEY_GIVEN_LANES_DESCRIPTION,
    MODE_CYCLING,
)


OPTIMIZE_PARAMS = {
    "bike_flow_constant": 1,
    "car_flow_constant": 1,
    "optimize_every_x": 1000,  # TODO: set lower for final version
    "valid_edges_k": 0,
    "car_weight": 2,
    "sp_method": "od",
    "shared_lane_factor": 2,
    "weight_od_flow": False,
}
IGNORE_FIXED = False


def generate_motorized_lane_graph(
    edge_path,
    node_path,
    source_lanes_attribute=KEY_LANES_DESCRIPTION,
    target_lanes_attribute=KEY_LANES_DESCRIPTION_AFTER,
    return_H=False,
):
    """SNMan way to load a street graph and convert it into a lane graph"""
    G = io.load_street_graph(edge_path, node_path)  # initialize lanes after rebuild
    print("Initial street graph edges:", G.number_of_edges())
    # need to save the maxspeed attribute here to use it later
    nx.set_edge_attributes(G, nx.get_edge_attributes(G, source_lanes_attribute), target_lanes_attribute)
    # ensure consistent edge directions (only from lower to higher node!)
    street_graph.organize_edge_directions(G)

    # # for using multi_set_given:
    # if len(nx.get_edge_attributes(G, "grade")) == 0:
    #     nx.set_edge_attributes(G, 0, "grade")
    # rebuilding.multi_set_given_lanes(G)
    distribution.set_given_lanes(G)
    H = street_graph.filter_lanes_by_modes(G, {MODE_PRIVATE_CARS}, lane_description_key=KEY_GIVEN_LANES_DESCRIPTION)

    merge_edges.reset_intermediate_nodes(H)
    merge_edges.merge_consecutive_edges(H, distinction_attributes={KEY_LANES_DESCRIPTION_AFTER})
    # make lane graph
    L = lane_graph.create_lane_graph(H, KEY_GIVEN_LANES_DESCRIPTION)
    print("Initial lane graph edges:", L.number_of_edges())
    # make sure that the graph is strongly connected
    L = graph.keep_only_the_largest_connected_component(L)
    print("Lane graph edges after keeping connected component:", L.number_of_edges())
    # add some edge attributes that we need for the optimization (e.g. slope)
    L = adapt_edge_attributes(L, ignore_fixed=IGNORE_FIXED)
    if return_H:
        return H, L
    return L


def adapt_edge_attributes(L: nx.MultiDiGraph, ignore_fixed=False):
    """Sets new edge attributes to be used in the optimization algorithm"""

    # rename to distance
    distances = nx.get_edge_attributes(L, "length")
    distances = {k: v / 1000 for k, v in distances.items()}  # transform to km
    nx.set_edge_attributes(L, distances, "distance")

    if ignore_fixed:
        nx.set_edge_attributes(L, False, "fixed")

    # add capacity and gradient attribute
    elevation_dict = nx.get_node_attributes(L, "elevation")  # can be empty, than flat assumed
    length_dict = nx.get_edge_attributes(L, "length")
    (gradient_attr, capacity_attr) = ({}, {})
    for u, v, k, d in L.edges(data=True, keys=True):
        e = (u, v, k)  # returning u, v, key, data
        gradient_attr[e] = 100 * (elevation_dict.get(v, 0) - elevation_dict.get(u, 0)) / length_dict[e]
        capacity_attr[e] = 0.5 if "P" in d["lanetype"] else 1
    nx.set_edge_attributes(L, capacity_attr, name="capacity")
    nx.set_edge_attributes(L, gradient_attr, name="gradient")

    # add location to node coordinates
    x_dict, y_dict = nx.get_node_attributes(L, "x"), nx.get_node_attributes(L, "y")
    loc = {n: {"loc": np.array([x_dict[n], y_dict[n]])} for n in L.nodes()}
    nx.set_node_attributes(L, loc)

    # add speed limit attribute
    speed_limit_attr = nx.get_edge_attributes(L, "maxspeed")
    # replace NaNs
    speed_limit_attr = {k: v if not pd.isna(v) else 30 for k, v in speed_limit_attr.items()}
    nx.set_edge_attributes(L, speed_limit_attr, name="speed_limit")

    return L


def lane_optimization(
    G_lane,
    od_df: pd.DataFrame = None,
    edge_fraction: float = 0.4,
    optimize_params: dict = OPTIMIZE_PARAMS,
    fix_multilane: bool = True,
) -> nx.MultiDiGraph:
    """Takes a lane graph from the city of Zurich, creates the OD matrix, and runs optimization"""

    if od_df is not None:
        od = od_df.copy()
    else:
        # Initialize od with empty dataframe
        od = pd.DataFrame(columns=["s", "t", "trips"])
    # extend OD matrix because otherwise we get disconnected car graph
    od = extend_od_circular(od, list(G_lane.nodes()))
    if od_df is None:
        od["trips"] = 1  # if all weightings are 0, it doesn't work, so we have to set it to 1 in this case

    print(f"Optimizing lane graph, {G_lane.number_of_edges()} edges and {G_lane.number_of_nodes()} nodes")
    print(f"with {len(od)} OD pairs")

    # run Pareto frontier
    opt = ParetoRoundOptimize(G_lane.copy(), od.copy(), **optimize_params)
    optimized_G_lane = opt.allocate_x_bike_lanes(fraction_bike_lanes=edge_fraction, fix_multilane=fix_multilane)
    return optimized_G_lane


def lane_optimization_snman(
    L,
    L_existing=None,
    street_graph=None,
    width_attribute=None,
    od_df_path="../street_network_data/birchplatz/raw_od_matrix/od_whole_city.csv",
    edge_fraction=0.4,
    optimize_params=OPTIMIZE_PARAMS,
    verbose=True,
    crs=2056,
):
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
    G_lane = adapt_edge_attributes(L)
    G_lane.remove_edges_from(nx.selfloop_edges(G_lane))

    # make OD matrix
    if os.path.exists(od_df_path):
        node_gdf = nodes_to_geodataframe(G_lane, crs=crs)
        od_df = match_od_with_nodes(station_data_path=od_df_path, nodes=node_gdf)
    else:
        od_df = None
        warnings.warn("Attention: The path to a city-wide OD-matrix does not exist, so we are using a random OD")

    new_lane_graph = lane_optimization(
        G_lane, od_df=od_df, edge_fraction=edge_fraction, optimize_params=optimize_params
    )

    # return only the car lanes
    G_filtered = filter_by_attribute(new_lane_graph, "lanetype", "M>")
    print(G_filtered.number_of_edges(), G_filtered.number_of_nodes())

    # delete loc
    remove_node_attribute(G_filtered, "loc")

    return G_filtered
