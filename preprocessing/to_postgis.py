import os
import numpy as np
import pandas as pd
import geopandas as gpd
import time
from shapely import wkt
import datetime
import networkx as nx
from shapely.geometry import Point
import geopandas as gpd
import shapely.ops

from ebike_city_tools.graph_utils import street_to_lane_graph
from ebike_city_tools.app_utils import get_database_connector

DATABASE_CONNECTOR = get_database_connector()


def join_with_geometry(edges, edges_geom):
    """Joins the edges with their detailed geometry"""

    def reverse_geom(geom):
        def _reverse(x, y, z=None):
            if z:
                return x[::-1], y[::-1], z[::-1]
            return x[::-1], y[::-1]

        return shapely.ops.transform(_reverse, geom)

    # drop duplicates
    edges_geom.drop_duplicates(subset=["u", "v"], inplace=True)

    # first, we create the reverse geometries before we can join them
    edges_geom_reversed = edges_geom.copy()
    edges_geom_reversed["v_temp"] = edges_geom_reversed["v"]
    edges_geom_reversed["v"] = edges_geom_reversed["u"]
    edges_geom_reversed["u"] = edges_geom_reversed["v_temp"]
    edges_geom_reversed.drop("v_temp", axis=1, inplace=True)
    len(edges_geom_reversed)
    edges_geom_reversed["geometry"] = edges_geom_reversed.geometry.apply(reverse_geom)
    # combined forward with backward
    edges_geom_all = pd.concat([edges_geom, edges_geom_reversed])

    edges_w_geom = pd.merge(edges, edges_geom_all, how="left", left_on=["source", "target"], right_on=["u", "v"])
    # keep only the source, target columns, drop u v
    edges_w_geom.drop(["u", "v"], axis=1, inplace=True)
    # add geometry
    edges_w_geom = gpd.GeoDataFrame(edges_w_geom, geometry="geometry")
    return edges_w_geom


def add_single_lane_flag(inp_edges):
    car_lanes = inp_edges[inp_edges["lanetype"].str.contains("M")]
    bike_lanes = inp_edges[~inp_edges["lanetype"].str.contains("M")]
    # for bike lanes, we don't need the chevrons anyways
    bike_lanes["flag"] = 0

    # add flag to car lanes
    car_lanes.set_index(["source", "target"], inplace=True)
    flag = []
    for s, t in car_lanes.index:
        if (t, s) in car_lanes.index:
            flag.append(0)
        else:
            flag.append(1)
    car_lanes["flag"] = flag
    new_all_lanes = pd.concat([bike_lanes, car_lanes.reset_index()])
    return new_all_lanes


def whole_city_graph_to_postgis(
    path_input="../street_network_data/zurich/street_graph_nodes.gpkg",
    path_output="outputs/rebuild_zurich/optimization_10_zurich/rebuild_whole_graph.gpkg",
):
    rebuild_output = gpd.read_file(path_output)
    rebuild_nodes = gpd.read_file(path_input)
    rebuild_output["ln_desc"] = rebuild_output["ln_desc_after"]
    print("bike lanes in rebuild graph", sum(rebuild_output["ln_desc_after"].str.contains("P")))

    include_lanetypes = ["H>", "H<", "M>", "M<", "M-", "T>", "<T", "P-", "L>", "<L"]
    rebuild_output["ln_desc"] = rebuild_output["ln_desc"].str.replace("P", "P-")
    lane_graph = street_to_lane_graph(
        rebuild_nodes.set_index("osmid").to_crs(2056),
        rebuild_output.set_index(["u", "v"]),
        include_lanetypes=include_lanetypes,
    )

    lane_gdf = nx.to_pandas_edgelist(lane_graph, edge_key="key")
    print("lane types in lane graph", lane_gdf["lanetype"].unique())
    lane_with_geometry = join_with_geometry(lane_gdf, rebuild_output[["u", "v", "geometry"]])

    # group by
    group_attrs = ["source", "target", "lanetype"]
    agg_dict = {attr: "first" for attr in lane_with_geometry.columns if attr not in group_attrs}
    agg_dict.update({"lanetype": "count"})
    save_graph = (
        lane_with_geometry.groupby(group_attrs).agg(agg_dict).rename({"lanetype": "count"}, axis=1).reset_index()
    )
    save_graph = gpd.GeoDataFrame(save_graph, geometry="geometry", crs=rebuild_output.crs)

    print(save_graph)
    # to postgis
    lane_with_geometry.to_postgis(
        "zurich_rebuild", DATABASE_CONNECTOR, schema="graphs", if_exists="replace", index=False
    )
    print("Written lane graph to postgis", len(lane_with_geometry))


include_attributes = [
    "source",
    "target",
    "edge_key",
    "fixed",
    "lanetype",
    "distance",
    "gradient",
    "speed_limit",
    "geometry",
]


def write_all_graphs_to_postgis():
    IN_PATH_DATA = "../street_network_data/"
    IN_PATH_OUTPUTS = "outputs/cluster_mylanegraph_final/cluster_final_optimized"
    for instance in os.listdir(IN_PATH_OUTPUTS):
        if instance[0] == ".":
            continue
        # 2) Process nodes
        nodes = gpd.read_file(os.path.join(IN_PATH_DATA, instance, "nodes_all_attributes.gpkg"))
        save_nodes = nodes.rename({"osmid": "node"}, axis=1).drop(
            ["traffic_signals", "osmid_original", "highway"], axis=1
        )
        save_nodes.to_postgis(
            f"{instance}_nodes", DATABASE_CONNECTOR, schema="graphs", if_exists="replace", index=False
        )

        # load edge geometries (same as lane_geometries.gpkg)
        inst_short = instance[:-2] if "_1" in instance or "_2" in instance else instance

        edge_geometries = gpd.read_file(os.path.join(IN_PATH_DATA, inst_short, "edges_all_attributes.gpkg"))

        for f in os.listdir(os.path.join(IN_PATH_OUTPUTS, instance)):
            if not "csv" in f or not "graph" in f:
                continue

            # get number of edges allocated
            params_from_name = f[:-4].split("_")
            edges_allocated, car_weight = int(params_from_name[-1]), params_from_name[-4]
            try:
                if float(car_weight) >= 1:
                    car_weight = str(int(float(car_weight)))
                algorithm = f"carweight{car_weight}"
            except ValueError:
                car_weight = "1"
                if "cartime" in f:
                    algorithm = "carbetweenness"
                else:
                    algorithm = "betweenness"

            # RESTRICT HERE
            if (
                edges_allocated not in [100, 200]
                or "topdown" in f
                # or "cartime" in f
                or car_weight not in ["1", "4", "8"]
            ):
                continue
            # print("processing", instance, f)

            edges = pd.read_csv(os.path.join(IN_PATH_OUTPUTS, instance, f))

            # join edges with geometry
            edges_w_geom = join_with_geometry(edges, edge_geometries)

            # reduce to the necessary attributes
            save_graph = edges_w_geom[include_attributes]
            save_graph = save_graph[save_graph.geometry.is_valid]
            save_graph["lanetype"] = save_graph["lanetype"].map({"P": "P", "M>": "M", "M": "M", "H": "M"})
            # filter out bike-reverse lanes since we only want to plot it once
            save_graph = save_graph[~save_graph["edge_key"].str.contains("revbike")]

            # aggregate by lane and count number of occurences
            group_attrs = ["source", "target", "lanetype"]
            agg_dict = {attr: "first" for attr in include_attributes if attr not in group_attrs}
            agg_dict.update({"lanetype": "count"})
            save_graph = (
                save_graph.groupby(group_attrs).agg(agg_dict).rename({"lanetype": "count"}, axis=1).reset_index()
            )
            save_graph = gpd.GeoDataFrame(save_graph, geometry="geometry", crs=edge_geometries.crs)

            # add flag regarding double lanes
            save_graph = add_single_lane_flag(save_graph)

            out_name = f"edges_{algorithm}_bikelanes{edges_allocated}"
            save_graph.to_postgis(
                f"{instance}_{out_name}", DATABASE_CONNECTOR, schema="graphs", if_exists="replace", index=False
            )
            print("Written graph to database", f"{instance}_{out_name}")


if __name__ == "__main__":
    write_all_graphs_to_postgis()
