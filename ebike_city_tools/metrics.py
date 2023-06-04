import networkx as nx
import numpy as np
import math

# metrics for a directed graph
def sp_reachability(G):
    nr_nodes = G.number_of_nodes()
    node_reachable_ratio = []
    for _, sp_node_dict in nx.all_pairs_shortest_path(G):
        node_reachable_ratio.append(len(sp_node_dict.keys()) / nr_nodes)
    return np.mean(node_reachable_ratio)


def sp_length(G):
    sp_lens = []
    for _, sp_node_dict in nx.all_pairs_shortest_path(G):
        reachable_lens = [len(val) for _, val in sp_node_dict.items()]
        sp_lens.extend(reachable_lens)
    return np.mean(sp_lens)


def centrality_metrics(G, weights=None):
    metric_dict = {}
    metric_dict["closeness"] = np.mean(list(nx.closeness_centrality(G).values()))
    metric_dict["betweenness"] = np.mean(list(nx.betweenness_centrality(G).values()))
    metric_dict["degree"] = np.mean(list(nx.degree_centrality(G).values()))
    return metric_dict

# def centrality_undirected(G):
#     metric_dict = {}
#     metric_dict["current_flow_closeness"] = np.mean(
#         list(nx.current_flow_closeness_centrality(G, weight="weight").values())
#     )
#     metric_dict["current_flow_betweenness"] = np.mean(
#         list(nx.current_flow_betweenness_centrality(G, weight="weight").values())
#     )
#     return metric_dict
# TODO
# Centrality degrading of nodes that are central in the original graph

def apply_weight(value, weights, edge):
    if weights:
        return value * weights[edge]
    else:
        return value


def average_slope(G, bike_graph, weights=None):
    network_slopes = []
    for edge in bike_graph.edges():
        # need to be in the same units, e.g. metres.
        node_z_dif = abs(G.nodes[edge[0]]['elevation'] - G.nodes[edge[1]]['elevation'])
        node_xy_dif = G.edges[edge]['length']
        slope = node_z_dif / node_xy_dif
        network_slopes.append(apply_weight(slope, weights, edge))
    return np.median(network_slopes)


def average_euclidean_edge_length(G, bike_graph, weights=None):
    edge_lengths = []
    for edge in bike_graph.edges():
        edge_length = G.edges[edge]['length']
        edge_lengths.append(apply_weight(edge_length, weights, edge))
    return np.mean(edge_lengths)


def average_edge_linearity(G, bike_graph, weights=None):
    edges_lin_list = []

    for edge in bike_graph.edges():
        edge_length = G.edges[edge]['length']
        node_1 = (G.nodes[edge[0]]['loc'][0], G.nodes[edge[0]]['loc'][1])
        node_2 = (G.nodes[edge[1]]['loc'][0], G.nodes[edge[1]]['loc'][1])
        edge_distance = math.dist(node_1, node_2)
        edge_linearity = edge_distance / edge_length
        edges_lin_list.append(apply_weight(edge_linearity, weights, edge))
    return np.mean(edges_lin_list)


def average_sp_angular_shift(G, bike_graph):
    median_sps_angle_shift = []
    for node_sps in dict(nx.all_pairs_shortest_path(bike_graph)).values():
        for sp in node_sps.values():
            # angular shift only computed on shortest paths with at least 2edges/3nodes.
            if len(sp) < 3:
                continue
            sp_angles = []
            for i, node in enumerate(sp[:-2]):
                node_1 = np.array([G.nodes[node]['loc'][0], G.nodes[node]['loc'][1]])
                node_2 = np.array([G.nodes[sp[i + 1]]['loc'][0], G.nodes[sp[i + 1]]['loc'][1]])
                node_3 = np.array([G.nodes[sp[i + 2]]['loc'][0], G.nodes[sp[i + 2]]['loc'][1]])
                edge_v1 = node_1 - node_2
                edge_v2 = node_3 - node_2
                cosine = np.dot(edge_v1, edge_v2) / (np.linalg.norm(edge_v1) * np.linalg.norm(edge_v2))
                angle = math.degrees(math.acos(cosine))
                sp_angles.append(180 - abs(angle))
            median_sps_angle_shift.append(np.mean(sp_angles))
    return np.mean(median_sps_angle_shift)
