import networkx as nx
import pandas as pd

from ebike_city_tools.optimize.linear_program import define_IP
from ebike_city_tools.utils import (
    lane_to_street_graph,
    extend_od_circular,
    output_lane_graph,
    filter_by_attribute,
    output_to_dataframe,
)
from ebike_city_tools.optimize.round_simple import rounding_and_splitting

WEIGHT_OD_FLOW = False
FLOW_CONSTANT = 1


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
    # isna = 0
    for e in L.edges:
        (u, v, _) = e  # TODO: remove loops
        gradient_attr[e] = 100 * (elevation_dict.get(v, 0) - elevation_dict.get(u, 0)) / length_dict[e]
        # Old version for speed limit (from G instead of L)
        # set speed limit attribute
        # if (u, v, 0) in maxspeed.keys():
        #     speed_limit_attr[e] = maxspeed[(u, v, 0)]
        # elif (v, u, 0) in maxspeed.keys():
        #     speed_limit_attr[e] = maxspeed[(v, u, 0)]
        # else:
        #     speed_limit_attr[e] = pd.NA
        # if pd.isna(speed_limit_attr[e]):
        #     isna += 1
        #     speed_limit_attr[e] = 30
    speed_limit_attr = nx.get_edge_attributes(L, "maxspeed")
    # replace NaNs
    speed_limit_attr = {k: v if not pd.isna(v) else 30 for k, v in speed_limit_attr.items()}
    nx.set_edge_attributes(L, speed_limit_attr, name="speed_limit")
    nx.set_edge_attributes(L, gradient_attr, name="gradient")

    return L


def lane_optimization(L, od_df=None, edge_fraction=0.4, shared_lane_factor=2, verbose=True):
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

    for u, v, d in G_lane.edges(data=True):
        if pd.isna(d["gradient"]):
            print("lane NAN")

    if od_df is not None:
        od = od_df.copy()
    else:
        # Initialize od with empty dataframe
        od = pd.DataFrame(columns=["s", "t", "trips"])
    # reduce OD matrix to nodes that are in G_lane
    node_list = list(G_lane.nodes())
    od = od[(od["s"].isin(node_list)) & (od["t"].isin(node_list))]
    # extend OD matrix because otherwise we get disconnected car graph
    od = extend_od_circular(od, node_list)

    # change representation
    G_street = lane_to_street_graph(G_lane)

    for u, v, d in G_street.edges(data=True):
        if pd.isna(d["gradient"]):
            print("street NAN")

    ip = define_IP(
        G_street,
        cap_factor=1,
        car_weight=10,
        od_df=od,
        bike_flow_constant=FLOW_CONSTANT,
        car_flow_constant=FLOW_CONSTANT,
        shared_lane_factor=shared_lane_factor,
        weight_od_flow=WEIGHT_OD_FLOW,
    )
    ip.optimize()
    capacity_values = output_to_dataframe(ip, G_street)
    del ip
    # print("HERE", sum(capacity_values["u_b(e)"] > 0))

    # compute how many bike edges we want to have
    bike_edges_to_add = int(edge_fraction * G_lane.number_of_edges())

    # rounding algorithm
    bike_G, car_G = rounding_and_splitting(capacity_values, bike_edges_to_add=bike_edges_to_add)

    # combine bike and car graph in a new lane graph with all relevant attributes
    new_lane_graph = output_lane_graph(G_lane, bike_G, car_G, shared_lane_factor)

    # return only the car lanes
    G_filtered = filter_by_attribute(new_lane_graph, "lane", "M>")

    return G_filtered
