import os
import numpy as np
import geopandas as gpd
import pandas as pd
import networkx as nx

from ebike_city_tools.utils import extend_od_circular
from run_real_data import generate_motorized_lane_graph

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
        path = os.path.join("..", "street_network_data", city)
        np.random.seed(42)  # random seed for extending the od matrix
        # generate lane graph with snman
        edge_len_initial = len(gpd.read_file(os.path.join(path, "edges_all_attributes.gpkg")))
        G_lane = generate_motorized_lane_graph(
            os.path.join(path, "edges_all_attributes.gpkg"), os.path.join(path, "nodes_all_attributes.gpkg")
        )

        # load OD
        od = pd.read_csv(os.path.join(path, "od_matrix.csv"))
        od.rename({"osmid_origin": "s", "osmid_destination": "t"}, inplace=True, axis=1)
        od = od[od["s"] != od["t"]]
        # reduce OD matrix to nodes that are in G_lane
        node_list = list(G_lane.nodes())
        od = od[(od["s"].isin(node_list)) & (od["t"].isin(node_list))]
        od_size_before_extend = len(od)

        od = extend_od_circular(od, node_list)

        res.append(
            {
                "name": city,
                "nodes": G_lane.number_of_nodes(),
                "edges_undir": edge_len_initial,
                "edges": G_lane.number_of_edges(),
                "od": od_size_before_extend,
                "od_after_extend": len(od),
                "multi-edges": G_lane.number_of_edges() - nx.DiGraph(G_lane).number_of_edges(),
            }
        )
    res = pd.DataFrame(res)
    res.to_csv("figures/data_info.csv", index=False)
