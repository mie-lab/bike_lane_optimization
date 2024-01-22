import argparse
import os
import snman
from snman.constants import *
from ebike_city_tools.optimize.wrapper import lane_optimization

PERIMETER = "_debug"

# Set these paths according to your own setup
parser = argparse.ArgumentParser()
parser.add_argument("-d", "--data_dir", type=str, default="../snman/examples/data_v2/")
parser.add_argument("-o", "--out_dir", type=str, default="outputs/rebuild_zurich")
args = parser.parse_args()

data_directory = args.data_dir
output_path = args.out_dir
os.makedirs(output_path, exist_ok=True)

inputs_path = os.path.join(data_directory, "inputs")
process_path = os.path.join(data_directory, "process", PERIMETER)
export_path = os.path.join(data_directory, "outputs", PERIMETER)

CRS_internal = 2056  # for Zurich
CRS_for_export = 4326

print("Load street graph")
G = snman.io.load_street_graph(
    os.path.join(process_path, "street_graph_edges.gpkg"),
    os.path.join(process_path, "street_graph_nodes.gpkg"),
    crs=CRS_internal,
)

print("Load perimeters")
perimeters_gdf = snman.io.load_perimeters(os.path.join(inputs_path, "perimeters", "perimeters.shp"), crs=CRS_internal)

print("Load rebuilding regions")
# Polygons that define which streets will be reorganized
rebuilding_regions_gdf = snman.io.load_rebuilding_regions(
    os.path.join(inputs_path, "rebuilding_regions", "rebuilding_regions.gpkg"), crs=CRS_internal
)

print("Load measurement regions")
# Polygons that define areas where network measures will be calculated
measurement_regions_gdf = snman.io.load_measurement_regions(
    os.path.join(inputs_path, "measurement_regions", "measurement_regions.gpkg"), crs=CRS_internal
)

print("Rebuild regions")
snman.rebuilding.rebuild_regions(
    G,
    rebuilding_regions_gdf,
    rebuilding_function=lane_optimization,
    verbose=True,
    export_G=output_path,
)
