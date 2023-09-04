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


def lane_to_street_graph(G_lane):
    # convert to dataframe
    G_dataframe = nx.to_pandas_edgelist(G_lane)
    G_dataframe = G_dataframe[["source", "target", "capacity", "gradient", "distance"]]
    assert all(G_dataframe["source"] != G_dataframe["target"])
    G_dataframe["source undir"] = G_dataframe[["source", "target"]].min(axis=1)
    G_dataframe["target undir"] = G_dataframe[["source", "target"]].max(axis=1)
    # G_dataframe.apply(lambda x: tuple(sorted([x["source"], x["target"]])), axis=1)
    # aggregate to get undirected
    undirected_edges = G_dataframe.groupby(["source undir", "target undir"]).agg(
        {"capacity": "sum", "distance": "first"}
    )

    # generate gradient first only for the ones where source < target
    grad_per_dir_edge = G_dataframe.groupby(["source", "target"])["gradient"].first().reset_index()
    grad_per_dir_edge = grad_per_dir_edge[grad_per_dir_edge["target"] > grad_per_dir_edge["source"]]
    undirected_edges["gradient"] = grad_per_dir_edge.set_index(["source", "target"]).to_dict()["gradient"]

    # make the same edges in the reverse direction -> gradient * (-1)
    reverse_edges = undirected_edges.reset_index()
    reverse_edges["source"] = reverse_edges["target undir"]
    reverse_edges["target"] = reverse_edges["source undir"]
    reverse_edges["gradient"] = reverse_edges["gradient"] * (-1)
    # concatenate
    directed_edges = pd.concat(
        [
            undirected_edges.reset_index().rename({"source undir": "source", "target undir": "target"}, axis=1),
            reverse_edges.drop(["source undir", "target undir"], axis=1),
        ]
    ).reset_index(drop=True)

    G_streets = nx.from_pandas_edgelist(
        directed_edges,
        source="source",
        target="target",
        edge_attr=["capacity", "distance", "gradient"],
        create_using=nx.DiGraph,
    )
    # set attributes (not really needed)
    # attrs = {i: {"loc": coords[i, :2], "elevation": coords[i, 2]} for i in range(len(coords))}
    # nx.set_node_attributes(G_streets, attrs)
    return G_streets


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
    return od_new.drop_duplicates()


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


def add_bike_and_car_time(G_lane, bike_G, car_G, shared_lane_factor=2):
    """
    Add two attributes to the graph G_lane: biketime and cartime
    """
    bike_edges = bike_G.edges()
    car_edges = car_G.edges()

    # thought: can keep it a multigraph because we are iterating
    for u, v, k, attributes in G_lane.edges(data=True, keys=True):
        e = (u, v, k)

        # 1) Car travel time: simply check whether the edge exists:
        if (u, v) in car_edges:
            G_lane.edges[e]["cartime"] = attributes["distance"] / 30
        else:
            G_lane.edges[e]["cartime"] = np.inf

        # 2) Bike travel time
        # Case 1: There is a bike lane on this edge
        if (u, v) in bike_edges:
            if attributes["gradient"] > 0:
                G_lane.edges[e]["biketime"] = attributes["distance"] / max(
                    [21.6 - 1.44 * attributes["gradient"], 1]
                )  # at least 1kmh speed
            else:  # if gradient < 0, than speed must increase --> - * -
                G_lane.edges[e]["biketime"] = attributes["distance"] / (21.6 - 0.86 * attributes["gradient"])
        # Case 2: There is no bike lane, but a car lane in the right direction --> multiply with shared_lane_factor
        elif (u, v) in car_edges:
            if attributes["gradient"] > 0:
                G_lane.edges[e]["biketime"] = (
                    shared_lane_factor * attributes["distance"] / max([21.6 - 1.44 * attributes["gradient"], 1])
                )  # at least 1kmh speed
            else:
                G_lane.edges[e]["biketime"] = (
                    shared_lane_factor * attributes["distance"] / (21.6 - 0.86 * attributes["gradient"])
                )
        # Case 3: Neither a bike lane nor a directed car lane exists
        else:
            G_lane.edges[e]["biketime"] = np.inf
    return G_lane
