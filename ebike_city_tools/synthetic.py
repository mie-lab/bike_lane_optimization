import networkx as nx
import numpy as np
import pandas as pd


def get_city_coords(n=20):
    coords = np.random.rand(n, 3)  # with elevation
    coords[:, :2] *= 5000  # positions vary between 0 and 5000m --> city of 5km quadtric side length
    coords[:, 2] *= 50  # make the altitude differ by at most 100m
    # --> NOTE: it is not ensured that nearby nodes have similar altitude, so we leave it like it is
    return coords.astype(int)


def make_fake_od(n, nr_routes, nodes=None):
    possible_pairs = np.array([[i, j] for i in range(n) for j in range(n) if i != j])
    selected_pair_inds = np.random.choice(np.arange(len(possible_pairs)), size=nr_routes, replace=False)
    selected_pairs = possible_pairs[selected_pair_inds]
    od = pd.DataFrame(selected_pairs, columns=["s", "t"])
    od["trips"] = (np.random.rand(nr_routes) * 5 + 1).astype(int)
    od = od[od["s"] != od["t"]].drop_duplicates(subset=["s", "t"])

    if nodes is not None:
        # transform into the correct node names
        node_list = np.array(sorted(list(nodes)))
        as_inds = od[["s", "t"]].values
        trips_per_day = od["trips"].values  # save flow column here
        # use as index
        od = pd.DataFrame(node_list[as_inds], columns=["s", "t"])
        od["trips"] = trips_per_day
    return od


def random_lane_graph(n=20, neighbor_choices=[2, 3, 4], neighbor_p=[0.6, 0.3, 0.1]):
    """
    Create realistic city graph with coordinates, elevation, etc
    Returns: MultiDiGraph with attributes width, distance, gradient -> one edge per lane!
    """
    # init graph
    G_lane = nx.MultiDiGraph()

    # define node coordinates
    coords = get_city_coords(n)
    node_inds = np.arange(n)
    node_ids = np.arange(n)  # * 10 # to test whether it works also for other node IDs

    # add edge list
    edge_list = []
    for i, node_coords in enumerate(coords):
        neighbor_distances = np.linalg.norm(coords[:, :2] - node_coords[:2], axis=1)
        neighbor_distances[neighbor_distances == 0] = 100000
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
            edge_list.append(
                [
                    node_ids[i],
                    node_ids[neigh],
                    {"capacity": 1, "distance": dist, "gradient": gradient, "speed_limit": 30},
                ]
            )
            edge_list.append(
                [
                    node_ids[neigh],
                    node_ids[i],
                    {"capacity": 1, "distance": dist, "gradient": -gradient, "speed_limit": 30},
                ]
            )
    G_lane.add_edges_from(edge_list)

    # set attributes
    attrs = {node_ids[i]: {"loc": coords[i]} for i in range(n)}
    nx.set_node_attributes(G_lane, attrs)

    if not nx.is_strongly_connected(G_lane):
        return random_lane_graph(n=n, neighbor_choices=neighbor_choices, neighbor_p=neighbor_p)

    return nx.MultiDiGraph(G_lane)
