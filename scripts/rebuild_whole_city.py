import argparse
import os
import geopandas as gpd
import pandas as pd
import osmnx
import networkx as nx
from ebike_city_tools.optimize.wrapper import lane_optimization, lane_optimization_snman
from ebike_city_tools.graph_utils import (
    load_lane_graph,
    nodes_to_geodataframe,
    keep_only_the_largest_connected_component,
)
from ebike_city_tools.utils import match_od_with_nodes
from snman.constants import *
import snman

PERIMETER = "_debug"
whole_city_od_path = "../street_network_data/birchplatz/raw_od_matrix/od_whole_city.csv"


def snman_rebuilding(data_directory, output_path):
    inputs_path = os.path.join(data_directory, "inputs")
    process_path = os.path.join(data_directory, "process", PERIMETER)
    export_path = os.path.join(data_directory, "outputs", PERIMETER)

    print("Load street graph")
    G = snman.io.load_street_graph(
        os.path.join(process_path, "street_graph_edges.gpkg"),
        os.path.join(process_path, "street_graph_nodes.gpkg"),
        crs=CRS_internal,
    )

    print("Load rebuilding regions")
    # Polygons that define which streets will be reorganized
    rebuilding_regions_gdf = snman.io.load_rebuilding_regions(
        os.path.join(inputs_path, "rebuilding_regions", "rebuilding_regions.gpkg"), crs=CRS_internal
    )

    print("Rebuild regions")
    snman.rebuilding.rebuild_regions(
        G,
        rebuilding_regions_gdf,
        rebuilding_function=lane_optimization_snman,
        verbose=True,
        export_G=output_path,
    )
    # save G after rebuilding is finished


def apply_changes_to_street_graph(street_graph_init: pd.DataFrame, G_lane_modified: pd.DataFrame):
    """
    Applies changes in the motorized lanes in a lane graph to the original big street graph
    """
    # get count per source-target-lane group
    count_per_lane = G_lane_modified.groupby(["source", "target", "lanetype"])["lane_id"].count()

    # iterate over edges in the original graph and modify where necessary
    for u, v in street_graph_init.index:
        # extract old lanes
        old_lanes = street_graph_init.loc[(u, v), "ln_desc"].replace("M-", "M> | M<").split(" | ")

        m_forward_count, m_backward_count, p_count = 0, 0, 0
        try:
            m_forward_count += count_per_lane.loc[u, v, "M>"]
        except KeyError:
            pass
        try:
            m_backward_count += count_per_lane.loc[v, u, "M>"]
        except KeyError:
            pass
        try:
            p_count += count_per_lane.loc[u, v, "P"]
            assert count_per_lane.loc[v, u, "P"] > 0
        except KeyError:
            pass

        # no bike lane was placed, so we don't need to change anything at that edge?
        # no, the direction of the car lane might have changed nevertheless
        #     if p_count == 0:
        #         continue

        new_lanes = (
            ["M>" for _ in range(m_forward_count)]
            + ["M<" for _ in range(m_backward_count)]
            + ["P" for _ in range(p_count)]
        )
        if len(new_lanes) == 0:
            # no changes at this edge or it wasn't even processed
            continue

        # there are some new lanes
        old_other_lanes = [l for l in old_lanes if "M" not in l]
        old_motorized_lanes = [l for l in old_lanes if "M" in l]

        # if there has been any change at all, update the motorized edges
        if not sorted(new_lanes) == sorted(old_motorized_lanes):
            # if a bike lane was placed, delete existing bike lanes
            if "P" in new_lanes:
                old_other_lanes = [l for l in old_other_lanes if "P" not in l and "L" not in l]
            new_lanes_string = " | ".join(old_other_lanes + new_lanes)
            print("OLD:", old_other_lanes, old_motorized_lanes, "NEW:", new_lanes, "ATTR:", new_lanes_string)
            street_graph_init.loc[(u, v), "ln_desc_after"] = new_lanes_string
    return street_graph_init


def rebuild_street_network(data_directory: str, output_path: str, out_attr_name: str = "ln_desc_after"):
    """
    Own implementation of rebuilding
    """
    # load lane graph - MultiDiGraph
    graph_path = os.path.join(data_directory, "process", PERIMETER)
    G_lane = load_lane_graph(
        graph_path, edge_fn="street_graph_edges.gpkg", node_fn="street_graph_nodes.gpkg", target_crs=CRS_internal
    )

    # load also the raw edges for saving in the same format in the end
    street_graph_edges = (
        gpd.read_file(os.path.join(graph_path, "street_graph_edges.gpkg")).set_index(["u", "v"]).to_crs(CRS_internal)
    )

    # initialize with the original lanes
    street_graph_edges[out_attr_name] = street_graph_edges["ln_desc"]

    rebuilding_regions_gdf = gpd.read_file(
        os.path.join(data_directory, "inputs", "rebuilding_regions", "rebuilding_regions.gpkg")
    ).to_crs(G_lane.graph["crs"])

    for i, rebuilding_region in rebuilding_regions_gdf.iterrows():
        print("\n--------\nRebuilding region", i, rebuilding_region["description"])
        # get the region polygon
        polygon = rebuilding_region["geometry"]

        # make a graph cutout based on the region geometry and skip this region if the resulting subgraph is empty
        G_lane_region = osmnx.truncate.truncate_graph_polygon(G_lane, polygon, quadrat_width=100, retain_all=True)
        if len(G_lane_region.edges) == 0:
            continue

        G_lane_region = keep_only_the_largest_connected_component(G_lane_region)
        print("After connected component", G_lane_region.number_of_nodes(), G_lane_region.number_of_edges())

        # make OD matrix for this region
        node_gdf = nodes_to_geodataframe(G_lane_region, crs=CRS_internal)
        od_df = match_od_with_nodes(station_data_path=whole_city_od_path, nodes=node_gdf)

        # optimize region
        optimized_G_lane = lane_optimization(
            G_lane_region,
            od_df,
        )

        # apply changes
        street_graph_edges = apply_changes_to_street_graph(
            street_graph_edges, nx.to_pandas_edgelist(optimized_G_lane, edge_key="key")
        )

        # save intermediate result
        street_graph_edges[[out_attr_name]].to_csv(
            os.path.join(output_path, f"rebuild_region_{i}_graph.csv"), index=True
        )

    # save final result
    street_graph_edges.to_csv(os.path.join(output_path, "rebuild_full_graph.csv"), index=True)


CRS_internal = 2056  # for Zurich
CRS_for_export = 4326

# Set these paths according to your own setup
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data_dir", type=str, default="../snman/examples/data_v2")
    parser.add_argument("-o", "--out_dir", type=str, default="outputs/rebuild_zurich")
    args = parser.parse_args()

    data_dir = args.data_dir
    output_dir = args.out_dir
    os.makedirs(output_dir, exist_ok=True)

    # snman_rebuilding(data_dir, output_dir)

    rebuild_street_network(data_dir, output_dir)
