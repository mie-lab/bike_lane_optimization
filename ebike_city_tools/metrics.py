import networkx as nx
import numpy as np
import pandas as pd
import warnings

from ebike_city_tools.utils import output_lane_graph


# metrics for a directed graph
def sp_reachability(G):
    nr_nodes = G.number_of_nodes()
    node_reachable_ratio = []
    for _, sp_node_dict in nx.all_pairs_shortest_path(G):
        node_reachable_ratio.append(len(sp_node_dict.keys()) / nr_nodes)
    return np.mean(node_reachable_ratio)


def sp_hops(G):
    sp_lens = []
    for _, sp_node_dict in nx.all_pairs_shortest_path(G):
        reachable_lens = [len(val) for _, val in sp_node_dict.items()]
        sp_lens.extend(reachable_lens)
    return np.mean(sp_lens)


def sp_length(G, attr="cartime", return_matrix=False):
    out = pd.DataFrame(nx.floyd_warshall(G, weight=attr))
    if return_matrix:
        return out.values
    return np.mean(out.values)


def closeness(G):
    return np.mean(list(nx.closeness_centrality(G).values()))


def od_sp(G, od, weight, weight_od_flow=False):
    """
    Compute shortest paths of the OD matrix, potentially weighted by the flow
    G: graph with edge attribute <weight>
    od: pd.DataFrame with columns s, t, row
    weight: str, name of the column with the attribute to minimize
    """
    sp = []
    for _, row in od.iterrows():
        # skip paths with weight = 0
        if row["trips_per_day"] == 0:
            continue
        # compute shortest path
        sp_len = nx.shortest_path_length(G, source=row["s"], target=row["t"], weight=weight)
        # apply weight if desired
        if weight_od_flow:
            sp_len *= row["trips_per_day"]
        sp.append(sp_len)
    return np.mean(sp)


# def centrality_metrics(G):
#     metric_dict = {}
#     metric_dict["closeness"] = np.mean(list(nx.closeness_centrality(G).values()))
#     # metric_dict["betweenness"] = np.mean(list(nx.betweenness_centrality(G).values()))
#     return metric_dict

# def centrality_undirected(G):
#     metric_dict = {}
#     metric_dict["current_flow_closeness"] = np.mean(
#         list(nx.current_flow_closeness_centrality(G, weight="weight").values())
#     )
#     metric_dict["current_flow_betweenness"] = np.mean(
#         list(nx.current_flow_betweenness_centrality(G, weight="weight").values())
#     )
#     return metric_dict


def compute_travel_times(
    G_lane, bike_G, car_G, od_matrix=None, sp_method="all_pairs", shared_lane_factor=2, weight_od_flow=False
):
    """
    Compute the travel times of bikes and car in the new lane graph composed of bike_G and car_G.
    Travel times are either computed over the whole graph (sp_method=all_pairs) or only with respect to the OD matrix
    if weight_od_flow=True, the travel times are weighted by the flow values
    """
    assert sp_method == "all_pairs" or od_matrix is not None, "if sp_method=od, an od matrix must be passed"
    assert bike_G.number_of_edges() + car_G.number_of_edges() == G_lane.number_of_edges()

    # transform the graph layout into travel times, including gradient and penalty factor for using
    # car lanes by bike
    G_lane_output = output_lane_graph(G_lane, bike_G, car_G, shared_lane_factor)
    # measure weighted times (floyd-warshall)
    if sp_method == "od":
        bike_travel_time = od_sp(G_lane_output, od_matrix, weight="biketime", weight_od_flow=weight_od_flow)
        car_travel_time = od_sp(G_lane_output, od_matrix, weight="cartime", weight_od_flow=weight_od_flow)
    else:
        bike_travel_time = np.mean(pd.DataFrame(nx.floyd_warshall(G_lane_output, weight="biketime")).values)
        car_travel_time = np.mean(pd.DataFrame(nx.floyd_warshall(G_lane_output, weight="cartime")).values)
    return {
        "bike_edges": bike_G.number_of_edges(),
        "car_edges": car_G.number_of_edges(),
        "bike_time": bike_travel_time,
        "car_time": car_travel_time,
    }


def reversed_hypervolume_indicator(data, ref_point=None):
    """Compute the hypervolume indicator (area with respect to a reference point) for a 2D pareto frontier"""
    if ref_point is None:
        ref_point = np.array(np.min(data, axis=0))
    last_ref_point = ref_point.copy()
    hypervolume = 0

    # reduce to points within ref point
    data_within_ref = data[(data[:, 0] >= ref_point[0]) & (data[:, 1] >= ref_point[1])]
    if len(data_within_ref) < len(data):
        warnings.warn("removed points below reference point")

    # sort by first and then by second column and drop duplicates
    data_df = pd.DataFrame(data_within_ref).drop_duplicates()
    data = data_df.sort_values([0, 1], ascending=[True, False]).values

    for i in range(len(data)):
        # make sure that the ordering is correct
        assert data[i, 0] >= last_ref_point[0], f"{data[i, 0]} and {last_ref_point[0]}"
        rectangle_sides = data[i] - last_ref_point
        # volume = area = side * side
        volume = rectangle_sides[0] * rectangle_sides[1]
        hypervolume += volume
        # set new ref point
        last_ref_point = np.array([data[i, 0], last_ref_point[1]])
    return hypervolume


def hypervolume_indicator(data, ref_point=None):
    """
    "Compute the hypervolume indicator (area with respect to a reference point) for a 2D pareto frontier
    """
    if ref_point is None:
        ref_point = np.array(np.max(data, axis=0))
    last_ref_point = ref_point.copy()

    hypervolume = 0

    # reduce to points within ref point
    data_within_ref = data[(data[:, 0] <= ref_point[0]) & (data[:, 1] <= ref_point[1])]
    if len(data_within_ref) < len(data):
        warnings.warn("removed points below reference point")

    # sort by first and then by second column and drop duplicates
    data_df = pd.DataFrame(data_within_ref).drop_duplicates()
    data = data_df.sort_values([0, 1], ascending=[True, False]).values

    for i in range(len(data)):
        # make sure that the ordering is correct
        #         assert data[i, 0] >= last_ref_point[0], f"{data[i, 0]} and {last_ref_point[0]}"
        rectangle_sides = last_ref_point - data[i]
        # volume = area = side * side
        volume = rectangle_sides[0] * rectangle_sides[1]
        hypervolume += volume
        # set new ref point
        last_ref_point = np.array([last_ref_point[0], data[i, 1]])

    ref_point_volume = ref_point[0] * ref_point[1]
    return ref_point_volume - hypervolume
