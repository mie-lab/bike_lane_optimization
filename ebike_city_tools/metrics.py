import networkx as nx
import numpy as np
import pandas as pd


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


def sp_length(G, attr="cartime"):
    out = pd.DataFrame(nx.floyd_warshall(G, weight=attr))
    return np.mean(out.values)


def closeness(G):
    return np.mean(list(nx.closeness_centrality(G).values()))


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
# TODO
# Centrality degrading of nodes that are central in the original graph
