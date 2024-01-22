import os
import geopandas as gpd
import numpy as np
import networkx as nx
from collections import defaultdict
import pandas as pd


def load_lane_graph(path: str, maxspeed_fill_val: int = 50):
    # street_graph_init # ?[]
    street_graph_edges = gpd.read_file(os.path.join(path, "edges_all_attributes.gpkg")).set_index(["u", "v"])
    street_graph_nodes = gpd.read_file(os.path.join(path, "nodes_all_attributes.gpkg")).set_index("osmid")

    # create node attributes
    if "elevation" in street_graph_nodes.columns:
        elevation = street_graph_nodes["elevation"].to_dict()
    else:
        elevation = defaultdict(int)
    node_attr = {
        idx: {"loc": np.array([row["x"], row["y"]]), "elevation": elevation[idx]}
        for idx, row in street_graph_nodes.iterrows()
    }
    # check if we have the max speed
    maxspeed_exists = "maxspeed" in street_graph_edges.columns

    # filter edges
    cap_per_lanetype = {"H": 1, "M": 1, "P": 0.5, "L": 0.5}
    include_lanetypes = ["H>", "H<", "M>", "M<", "M-"]
    fixed_lanetypes = ["H>", "<H"]

    lane_graph_rows = []
    for (u, v), row in street_graph_edges.iterrows():
        #     print("---", u,v, row["ln_desc"])
        for lt in row["ln_desc"].split(" | "):
            if lt not in include_lanetypes:
                continue
            # forward lane

            # extract lane properties
            cap = cap_per_lanetype.get(lt[0], 1)
            property_dict = {
                "lanetype": lt[0],
                "distance": row["length"] / 1000,
                "capacity": cap,
                "gradient": 100 * (elevation.get(v, 0) - elevation.get(u, 0)) / row["length"],
                "fixed": lt in fixed_lanetypes,
                "speed_limit": row["maxspeed"]
                if maxspeed_exists and not pd.isna(row["maxspeed"])
                else maxspeed_fill_val,
            }
            if ">" in lt or "-" in lt:
                # add lane in this direction
                property_dict.update({"u": u, "v": v})
                lane_graph_rows.append(property_dict.copy())
            #             print(lane_graph_rows[-1])
            if "<" in lt or "-" in lt:
                # add opposite direction edge
                property_dict.update({"u": v, "v": u})
                property_dict["gradient"] = property_dict["gradient"] * (-1)
                lane_graph_rows.append(property_dict.copy())
    #             print(lane_graph_rows[-1])

    lane_graph_rows = pd.DataFrame(lane_graph_rows)
    assert len(lane_graph_rows) == len(lane_graph_rows.dropna())

    attrs = [c for c in lane_graph_rows.columns if c not in ["u", "v"]]
    lane_graph = nx.from_pandas_edgelist(
        lane_graph_rows, edge_attr=attrs, source="u", target="v", create_using=nx.MultiDiGraph
    )
    nx.set_node_attributes(lane_graph, node_attr)
    return lane_graph


def lane_to_street_graph(g_lane):
    # transform into an undirected street graph
    g_street = nx.Graph(g_lane)
    caps_g_street = defaultdict(int)
    # iterate over the original edges and set capacities (same for forward and backward edge)
    for u, v, d in g_lane.edges(data=True):
        if u < v:
            caps_g_street[(u, v)] += d["capacity"]  # add the capacity of this lane
        else:
            caps_g_street[(v, u)] += d["capacity"]
    nx.set_edge_attributes(g_street, caps_g_street, "capacity")

    # now make it directed -> will automatically replace every edge by two
    g_street = g_street.to_directed()

    # get original gradients
    for u, v in g_street.edges():
        if (u, v) in g_lane.edges():
            g_street[u][v]["gradient"] = g_lane[u][v][list(g_lane[u][v].keys())[0]]["gradient"]
        # only the edge in the opposite direction existed in the original graph
        elif (v, u) in g_lane.edges():
            g_street[u][v]["gradient"] = -1 * g_lane[v][u][list(g_lane[v][u].keys())[0]]["gradient"]
        else:
            raise RuntimeError(
                "This should not happen, all edges in g_street should be in g_lane in one dir or the other"
            )
    assert nx.is_strongly_connected(g_street)
    return g_street


def remove_node_attribute(G, attr):
    """Sets this attribute to zero for all nodes"""
    nx.set_node_attributes(G, {n: 0 for n in G.nodes()}, attr)


def remove_edge_attribute(G, attr):
    """Sets this attribute to zero for all nodes"""
    nx.set_edge_attributes(G, {e: 0 for e in G.edges()}, attr)


def filter_by_attribute(G, attr, attr_value):
    """
    Create a subgraph of G consisting of the edges with attr=attr_value
    """
    edges_df = nx.to_pandas_edgelist(G)
    edges_df = edges_df[edges_df[attr] == attr_value]
    attr_cols = [c for c in edges_df.columns if c not in ["source", "target"]]
    G_filtered = nx.from_pandas_edgelist(
        edges_df, edge_attr=attr_cols, source="source", target="target", create_using=type(G)
    )
    # make sure that the other attributes are transferred to the new graph
    G_filtered = transfer_node_attributes(G, G_filtered)
    return G_filtered


def nodes_to_geodataframe(G, crs=4326):
    to_df = []
    for n, d in G.nodes(data=True):
        d_dict = {"osmid": n}
        d_dict.update(d)
        to_df.append(d_dict)
    gdf = gpd.GeoDataFrame(to_df, geometry="geometry", crs=crs)
    if crs == 4326 and abs(gdf.geometry.x.iloc[0]) > 200:
        raise RuntimeError("the crs seems to be wrong. Please set a suitable crs")
    return gdf


def transfer_node_attributes(G_in, G_out):
    for n, data in G_in.nodes(data=True):
        attr_keys = data.keys()
        break
    for a in attr_keys:
        attr_dict = nx.get_node_attributes(G_in, a)
        nx.set_node_attributes(G_out, attr_dict, name=a)
    # transfer graph
    for k in G_in.graph.keys():
        G_out.graph[k] = G_in.graph[k]
    return G_out


def determine_vertices_on_shortest_paths(od_pair, graph, number_of_paths):
    """Auxiliary routine to extract the vertices lying on the shortest path for the given od_pair."""
    graph_copy = graph.copy()
    s = od_pair[0]
    t = od_pair[1]
    vertices_on_shortest_paths = set()
    for _ in range(number_of_paths):
        try:
            path = nx.shortest_path(graph_copy, s, t, "bike_travel_time")
        except nx.exception.NetworkXNoPath:
            break
        vertices_on_shortest_paths.update(path)
        shortend_path = list(path[1:-1])
        graph_copy.remove_nodes_from(shortend_path)
    return vertices_on_shortest_paths


def determine_arcs_between_vertices(graph, vertices):
    """Auxiliary routine that returns all arcs in the graph, that have both endpoints in the specified vertex set."""
    valid_arcs = []
    for arc in graph.edges():
        if arc[0] in vertices and arc[1] in vertices:
            valid_arcs.append(arc)
    return list(set(valid_arcs))


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


def keep_only_the_largest_connected_component(G, weak=False):
    """
    TAKEN FROM SNMAN library
    Remove all nodes and edges that are disconnected from the largest connected component.
    For directed graphs, strong connectedness will be considered, unless weak=True

    Parameters
    ----------
    G : nx.MultiGraph
        street graph
    weak : bool
        use weakly connected component in case of a directed graph

    Returns
    -------
    H : copy of subgraph representing the largest connected component
    """

    if G.is_directed():
        if weak:
            nodes = max(nx.weakly_connected_components(G), key=len)
        else:
            nodes = max(nx.strongly_connected_components(G), key=len)
    else:
        nodes = max(nx.connected_components(G), key=len)

    return G.subgraph(nodes).copy()
