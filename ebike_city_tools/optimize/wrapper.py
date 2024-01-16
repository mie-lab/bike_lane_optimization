import networkx as nx
import pandas as pd
import numpy as np

from ebike_city_tools.optimize.linear_program import define_IP
from ebike_city_tools.utils import (
    lane_to_street_graph,
    extend_od_circular,
    output_lane_graph,
    filter_by_attribute,
    output_to_dataframe,
    remove_node_attribute,
    nodes_to_geodataframe,
    match_od_with_nodes,
)
from ebike_city_tools.optimize.round_optimized import ParetoRoundOptimize


OPTIMIZE_PARAMS = {
    "bike_flow_constant": 1,
    "car_flow_constant": 1,
    "optimize_every_x": 1000,  # TODO: set lower for final version
    "valid_edges_k": None,
    "car_weight": 2,
    "sp_method": "od",
    "shared_lane_factor": 2,
    "weight_od_flow": False,
}


def adapt_edge_attributes(L: nx.MultiDiGraph, ignore_fixed=False):
    """Sets new edge attributes to be used in the optimization algorithm"""
    # add capacity attribute to 1 for all lanes
    nx.set_edge_attributes(L, 1, name="capacity")

    # rename to distance
    distances = nx.get_edge_attributes(L, "length")
    distances = {k: v / 1000 for k, v in distances.items()}  # transform to km
    nx.set_edge_attributes(L, distances, "distance")

    if ignore_fixed:
        nx.set_edge_attributes(L, False, "fixed")

    # add gradient attribute
    elevation_dict = nx.get_node_attributes(L, "elevation")  # can be empty, than flat assumed
    length_dict = nx.get_edge_attributes(L, "length")
    gradient_attr, speed_limit_attr = {}, {}
    for e in L.edges:
        (u, v, _) = e  # TODO: remove loops
        gradient_attr[e] = 100 * (elevation_dict.get(v, 0) - elevation_dict.get(u, 0)) / length_dict[e]

    # add location to node coordinates
    x_dict, y_dict = nx.get_node_attributes(L, "x"), nx.get_node_attributes(L, "y")
    loc = {n: {"loc": np.array([x_dict[n], y_dict[n]])} for n in L.nodes()}
    nx.set_node_attributes(L, loc)

    # add speed limit attribute
    speed_limit_attr = nx.get_edge_attributes(L, "maxspeed")
    # replace NaNs
    speed_limit_attr = {k: v if not pd.isna(v) else 30 for k, v in speed_limit_attr.items()}
    nx.set_edge_attributes(L, speed_limit_attr, name="speed_limit")
    nx.set_edge_attributes(L, gradient_attr, name="gradient")

    return L


def lane_optimization(
    L,
    L_existing=None,
    street_graph=None,
    width_attribute=None,
    od_df=None,
    edge_fraction=0.1,
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
    node_gdf = nodes_to_geodataframe(G_lane, crs=crs)
    od_df = match_od_with_nodes(
        station_data_path="../street_network_data/birchplatz/raw_od_matrix/od_whole_city.csv", nodes=node_gdf
    )

    if od_df is not None:
        od = od_df.copy()
    else:
        # Initialize od with empty dataframe
        od = pd.DataFrame(columns=["s", "t", "trips"])
    # extend OD matrix because otherwise we get disconnected car graph
    od = extend_od_circular(od, list(G_lane.nodes()))
    if od_df is None:
        od["trips"] = 1  # if all weightings are 0, it doesn't work, so we have to set it to 1 in this case

    print(
        f"---------------\nProcessing lane graph, {G_lane.number_of_edges()} edges and {G_lane.number_of_nodes()} nodes"
    )
    print(f"with {len(od)} OD pairs")

    opt = ParetoRoundOptimize(G_lane.copy(), od.copy(), **optimize_params)
    num_bike_edges = int(edge_fraction * G_lane.number_of_edges())
    new_lane_graph = opt.pareto(return_graph=True, max_bike_edges=num_bike_edges)

    # return only the car lanes
    G_filtered = filter_by_attribute(new_lane_graph, "lanetype", "M>")
    print(G_filtered.number_of_edges(), G_filtered.number_of_nodes())

    # delete loc
    remove_node_attribute(G_filtered, "loc")

    return G_filtered
