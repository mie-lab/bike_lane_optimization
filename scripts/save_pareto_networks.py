import os
import numpy as np
import argparse
import networkx as nx
import pandas as pd
from ebike_city_tools.optimize.round_simple import rounding_and_splitting
from ebike_city_tools.utils import output_lane_graph
from ebike_city_tools.graph_utils import filter_by_attribute
from run_real_data import generate_motorized_lane_graph

from snman import distribution, street_graph, graph_utils, io, merge_edges, lane_graph
from snman.constants import *
from snman.rebuilding import rebuild_lanes_from_owtop_graph


def save_networks(H_in, G_lane_in, capacity_path, out_path, shared_lane_factor=2):
    """
    Load capacities from pre-saved and use it to create several lane graphs
    """
    capacity_values = pd.read_csv(capacity_path)
    # print("HERE", sum(capacity_values["u_b(e)"] > 0))

    for i, edge_fraction in enumerate(np.arange(0.05, 0.55, 0.05)):
        H = H_in.copy()
        G_lane = G_lane_in.copy()

        # compute how many bike edges we want to have
        bike_edges_to_add = int(edge_fraction * G_lane.number_of_edges())

        # assign bike edges based on the capacity values of the optimization approach
        bike_G, car_G = rounding_and_splitting(capacity_values, bike_edges_to_add=bike_edges_to_add)
        print(bike_G.number_of_edges())

        # combine bike and car graph in a new lane graph with all relevant attributes
        new_lane_graph = output_lane_graph(G_lane, bike_G, car_G, shared_lane_factor)

        # filter by car lanes (as in the link_elimination function, where only the leftover car lanes are returned)
        G_filtered = filter_by_attribute(new_lane_graph, "lane", "M>")

        # rebuild lanes in subgraph based on new lane graph
        rebuild_lanes_from_owtop_graph(
            H,
            G_filtered,
            source_lanes_attribute=KEY_LANES_DESCRIPTION_AFTER,
            target_lanes_attribute=KEY_LANES_DESCRIPTION_AFTER,
        )
        # reconstruct intermediary nodes and ensure consistent edge directions
        merge_edges.reconstruct_consecutive_edges(H)
        street_graph.organize_edge_directions(H)
        io.export_street_graph(H, os.path.join(out_path, f"edges_{i}.gpkg"), os.path.join(out_path, f"nodes_{i}.gpkg"))
        # write rebuilt lanes from subgraph into the main graph
        # nx.set_edge_attributes(G, nx.get_edge_attributes(H, KEY_LANES_DESCRIPTION_AFTER), KEY_LANES_DESCRIPTION_AFTER)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data_path", default="../street_network_data/birchplatz", type=str)
    parser.add_argument("-o", "--out_path", default="outputs/pareto_networks/", type=str)
    args = parser.parse_args()

    os.makedirs(args.out_path, exist_ok=True)
    path = args.data_path

    # generate lane graph with snman
    H, G_lane = generate_motorized_lane_graph(
        os.path.join(path, "edges_all_attributes.gpkg"),
        os.path.join(path, "nodes_all_attributes.gpkg"),
        return_H=True,
    )
    save_networks(H, G_lane, "../street_network_data/birchplatz/storymap_capacities.csv", out_path=args.out_path)
