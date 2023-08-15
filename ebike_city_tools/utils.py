import networkx as nx
import numpy as np
import pandas as pd


def lossless_to_undirected(graph):
    """
    Convert a directed multi graph into an undirected multigraph without loosing any edges
    Note: the networkx to_undirected() method converts each pair of reciprocal edges into a single undirected edge
    """
    edges = list(graph.edges(data=True))
    graph_unwo_edges = graph.copy()
    edges_to_remove = list(graph_unwo_edges.edges())
    graph_unwo_edges.remove_edges_from(edges_to_remove)
    graph_undir = nx.MultiGraph(graph_unwo_edges)
    graph_undir.add_edges_from(edges)
    return graph_undir


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
                np.stack([nodes_not_in_t[:min_len], shuffled_s]),
                np.stack([nodes_not_in_t[min_len:], shuffled_s[:len_diff]]),
            ],
            axis=1,
        )
    # transform to dataframe
    new_od_paths = pd.DataFrame(combined_nodes.swapaxes(1, 0), columns=["s", "t"])
    # concat and add a flow value of 1
    od_new = pd.concat([od, new_od_paths]).fillna(1)

    # check again
    nodes_not_in_s, nodes_not_in_t = get_missing_nodes(od_new)
    assert len(nodes_not_in_s) == 0 and len(nodes_not_in_t) == 0
    return od_new


def add_bike_and_car_time(G_city, bike_G, car_G, shared_lane_factor=2):
    """
    Add two attributes to the graph G_city: biketime and cartime
    """
    bike_edges = bike_G.edges()
    car_edges = car_G.edges()

    # thought: can keep it a multigraph because we are iterating
    for u, v, k, attributes in G_city.edges(data=True, keys=True):
        e = (u, v, k)

        # 1) Car travel time: simply check whether the edge exists:
        if (u, v) in car_edges:
            G_city.edges[e]["cartime"] = attributes["distance"] / 30
        else:
            G_city.edges[e]["cartime"] = np.inf

        # 2) Bike travel time
        # Case 1: There is a bike lane on this edge
        if (u, v) in bike_edges:
            if attributes["gradient"] > 0:
                G_city.edges[e]["biketime"] = attributes["distance"] / max(
                    [21.6 - 1.44 * attributes["gradient"], 1]
                )  # at least 1kmh speed
            else:  # if gradient < 0, than speed must increase --> - * -
                G_city.edges[e]["biketime"] = attributes["distance"] / (21.6 - 0.86 * attributes["gradient"])
        # Case 2: There is no bike lane, but a car lane in the right direction --> multiply with shared_lane_factor
        elif (u, v) in car_edges:
            if attributes["gradient"] > 0:
                G_city.edges[e]["biketime"] = (
                    shared_lane_factor * attributes["distance"] / max([21.6 - 1.44 * attributes["gradient"], 1])
                )  # at least 1kmh speed
            else:
                G_city.edges[e]["biketime"] = (
                    shared_lane_factor * attributes["distance"] / (21.6 - 0.86 * attributes["gradient"])
                )
        # Case 3: Neither a bike lane nor a directed car lane exists
        else:
            G_city.edges[e]["biketime"] = np.inf
    return G_city
