import os
import numpy as np
import geopandas as gpd
import pandas as pd
import networkx as nx

from ebike_city_tools.utils import extend_od_circular, fix_multilane_bike_lanes
from ebike_city_tools.graph_utils import load_lane_graph, keep_only_the_largest_connected_component

if __name__ == "__main__":
    res = []
    for city in [
        "affoltern",
        "birchplatz",
        "cambridge_1",
        "cambridge_2",
        "cambridge_3",
        "cambridge_4",
        "chicago_1",
        "chicago_2",
        "chicago",
    ]:
        print("getting infos for city", city)
        path = os.path.join("..", "street_network_data", city)
        np.random.seed(42)  # random seed for extending the od matrix
        # generate lane graph with snman
        edge_len_initial = len(gpd.read_file(os.path.join(path, "edges_all_attributes.gpkg")))

        G_lane = load_lane_graph(path)
        G_lane = keep_only_the_largest_connected_component(G_lane)

        # load OD
        od = pd.read_csv(os.path.join(path, "od_matrix.csv"))
        od.rename({"osmid_origin": "s", "osmid_destination": "t"}, inplace=True, axis=1)
        od = od[od["s"] != od["t"]]
        # reduce OD matrix to nodes that are in G_lane
        node_list = list(G_lane.nodes())
        od = od[(od["s"].isin(node_list)) & (od["t"].isin(node_list))]
        od_size_before_extend = len(od)

        od = extend_od_circular(od, node_list)

        multi_edges = len(fix_multilane_bike_lanes(G_lane, check_for_existing=False))

        res.append(
            {
                "name": city,
                "nodes": G_lane.number_of_nodes(),
                "edges_undir": edge_len_initial,
                "edges": G_lane.number_of_edges(),
                "od": od_size_before_extend,
                "od_after_extend": len(od),
                "multi-edges": multi_edges,
            }
        )
    res = pd.DataFrame(res)
    res.to_csv("figures/data_info.csv", index=False)
