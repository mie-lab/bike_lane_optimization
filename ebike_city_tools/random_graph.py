import networkx as nx
import numpy as np


def generate_base_graph(n=20, min_neighbors=2):
    """Random graph where the neighbors of each node are sampled proportinally to their inverse distance"""
    # init graph
    G = nx.MultiDiGraph()

    # define node coordinates
    coords = np.random.rand(n, 2)
    node_inds = np.arange(n)

    # add edge list
    edge_list = []
    for i, node_coords in enumerate(coords):
        neighbor_distances = np.linalg.norm(coords - node_coords, axis=1)
        neighbor_distances[i] = 1000000
        neighbor_probs = 1 / neighbor_distances ** 2
        neighbor_probs = neighbor_probs / np.sum(neighbor_probs)
        nr_neighbors = max(min_neighbors, round(np.random.normal(2.5, 2)))
        sampled_neighbors = np.random.choice(node_inds, p=neighbor_probs, replace=True, size=nr_neighbors)
        for neigh in sampled_neighbors:
            dist = neighbor_distances[neigh]
            edge_list.append([i, neigh, {"weight": np.random.rand(), "distance": dist}])
    #             edge_list.append([neigh, i, {"weight": np.random.rand()}])
    G.add_edges_from(edge_list)

    # set attributes
    attrs = {i: {"loc": coords[i]} for i in range(n)}
    nx.set_node_attributes(G, attrs)

    return G


def base_graph_doppelspur(n=20, min_neighbors=2):
    """
    Graph where each lane has also a corresponding lane in the opposite direction
    (no multigraph!)
    """
    # init graph
    G = nx.DiGraph()

    # define node coordinates
    coords = np.random.rand(n, 2)
    node_inds = np.arange(n)

    # add edge list
    edge_list = []
    for i, node_coords in enumerate(coords):
        neighbor_distances = np.linalg.norm(coords - node_coords, axis=1)
        neighbor_distances[i] = 1000000
        neighbor_probs = 1 / neighbor_distances ** 2
        neighbor_probs = neighbor_probs / np.sum(neighbor_probs)
        nr_neighbors = max(min_neighbors, round(np.random.normal(2, 2)))
        sampled_neighbors = np.random.choice(node_inds, p=neighbor_probs, replace=False, size=nr_neighbors)
        for neigh in sampled_neighbors:
            dist = neighbor_distances[neigh]
            edge_list.append([i, neigh, {"weight": np.random.rand(), "distance": dist}])
            edge_list.append([neigh, i, {"weight": np.random.rand(), "distance": dist}])
    G.add_edges_from(edge_list)

    # set attributes
    attrs = {i: {"loc": coords[i]} for i in range(n)}
    nx.set_node_attributes(G, attrs)

    return nx.MultiDiGraph(G)
