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
        sp_len = nx.shortest_path_length(G, source=row["s"], target=row["t"], weight=weight)
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
