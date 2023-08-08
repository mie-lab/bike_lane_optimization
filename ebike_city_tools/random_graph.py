import networkx as nx
import numpy as np
import pandas as pd


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
        neighbor_probs = 1 / neighbor_distances**2
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
        neighbor_probs = 1 / neighbor_distances**2
        neighbor_probs = neighbor_probs / np.sum(neighbor_probs)
        nr_neighbors = max(min_neighbors, round(np.random.normal(2, 2)))
        sampled_neighbors = np.random.choice(node_inds, p=neighbor_probs, replace=False, size=nr_neighbors)
        for neigh in sampled_neighbors:
            dist = neighbor_distances[neigh]
            edge_list.append([i, neigh, {"weight": 1, "distance": dist}])
            edge_list.append([neigh, i, {"weight": 1, "distance": dist}])
    G.add_edges_from(edge_list)

    # set attributes
    attrs = {i: {"loc": coords[i]} for i in range(n)}
    nx.set_node_attributes(G, attrs)

    return nx.MultiDiGraph(G)


def deprecated_aureliens_base_graph(n=20, min_neighbors=2):
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
        neighbor_probs = 1 / neighbor_distances**2
        neighbor_probs = neighbor_probs / np.sum(neighbor_probs)
        nr_neighbors = max(min_neighbors, round(np.random.normal(2, 2)))
        sampled_neighbors = np.random.choice(node_inds, p=neighbor_probs, replace=False, size=nr_neighbors)
        for neigh in sampled_neighbors:
            dist = neighbor_distances[neigh]
            cap = 10  # round(np.random.rand()*5) # TODO
            grad = np.random.rand() * 5  # Add to edges --> this is also done in the other one
            # they have the same capacity, so basically they are just virtual edges for the same edge
            edge_list.append([i, neigh, {"capacity": cap, "distance": dist, "gradient": grad}])
            edge_list.append([neigh, i, {"capacity": cap, "distance": dist, "gradient": -grad}])
    G.add_edges_from(edge_list)

    # set attributes
    attrs = {i: {"loc": coords[i]} for i in range(n)}
    nx.set_node_attributes(G, attrs)

    return nx.DiGraph(G)


def get_city_coords(n=20):
    coords = np.random.rand(n, 3)  # with elevation
    coords[:, :2] *= 5000  # positions vary between 0 and 5000m --> city of 5km quadtric side length
    coords[:, 2] *= 50  # make the altitude differ by at most 100m
    # --> NOTE: it is not ensured that nearby nodes have similar altitude, so we leave it like it is
    return coords.astype(int)


def city_graph(n=20, neighbor_choices=[2, 3, 4], neighbor_p=[0.6, 0.3, 0.1]):
    """
    Create realistic city graph with coordinates, elevation, etc
    Returns: MultiDiGraph with attributes width, distance, gradient -> one edge per lane!
    """
    # init graph
    G = nx.MultiDiGraph()

    # define node coordinates
    coords = get_city_coords(n)
    node_inds = np.arange(n)
    node_ids = np.arange(n)  # * 10 # to test whether it works also for other node IDs

    # add edge list
    edge_list = []
    for i, node_coords in enumerate(coords):
        neighbor_distances = np.linalg.norm(coords[:, :2] - node_coords[:2], axis=1)
        neighbor_distances[i] = 1000000
        neighbor_probs = 1 / neighbor_distances**2
        neighbor_probs = neighbor_probs / np.sum(neighbor_probs)
        # nr_neighbors = max(min_neighbors, round(np.random.normal(2, 2)))
        nr_neighbors = np.random.choice(neighbor_choices, p=neighbor_p)
        sampled_neighbors = np.random.choice(node_inds, p=neighbor_probs, replace=True, size=nr_neighbors)
        for neigh in sampled_neighbors:
            dist = neighbor_distances[neigh] / 1000  # we want the distance in km
            gradient = (coords[neigh, 2] - node_coords[2]) / (dist * 10)  # meter of height per 100m
            # gradient is given in percent
            # (From paper: for every additional 1% of uphill gradient,
            # the mean speed is reduced by 0.4002 m/s (1.44 kph))
            # --> meters in height / (dist * 1000) * 100
            edge_list.append([node_ids[i], node_ids[neigh], {"width": 1, "distance": dist, "gradient": gradient}])
            edge_list.append([node_ids[neigh], node_ids[i], {"width": 1, "distance": dist, "gradient": -gradient}])
    G.add_edges_from(edge_list)

    # set attributes
    attrs = {node_ids[i]: {"loc": coords[i]} for i in range(n)}
    nx.set_node_attributes(G, attrs)

    if not nx.is_strongly_connected(G):
        return city_graph(n=n, neighbor_choices=neighbor_choices, neighbor_p=neighbor_p)

    return nx.MultiDiGraph(G)


def lane_to_street_graph(G_city):
    # convert to dataframe
    G_dataframe = nx.to_pandas_edgelist(G_city)
    assert all(G_dataframe["source"] != G_dataframe["target"])
    G_dataframe["source undir"] = G_dataframe[["source", "target"]].min(axis=1)
    G_dataframe["target undir"] = G_dataframe[["source", "target"]].max(axis=1)
    # G_dataframe.apply(lambda x: tuple(sorted([x["source"], x["target"]])), axis=1)
    # aggregate to get undirected
    undirected_edges = G_dataframe.groupby(["source undir", "target undir"]).agg({"width": "sum", "distance": "first"})

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
    directed_edges = (
        pd.concat(
            [
                undirected_edges.reset_index().rename({"source undir": "source", "target undir": "target"}, axis=1),
                reverse_edges.drop(["source undir", "target undir"], axis=1),
            ]
        )
        .reset_index(drop=True)
        .rename({"width": "capacity"}, axis=1)
    )

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
