import networkx as nx
import numpy as np


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
