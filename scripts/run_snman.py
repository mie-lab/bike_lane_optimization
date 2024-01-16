import snman
from snman.constants import *
from ebike_city_tools.optimize.wrapper import lane_optimization

PERIMETER = "_debug"

# Set these paths according to your own setup
data_directory = "../snman/examples/data_v2/"
inputs_path = data_directory + "inputs/"
process_path = data_directory + "process/" + PERIMETER + "/"

# export_path = data_directory + 'outputs/' + 'matsim_zrh5 v5' + '/'
export_path = data_directory + "outputs/" + PERIMETER + "/"

CRS_internal = 2056  # for Zurich
CRS_for_export = 4326

print("Load street graph")
G = snman.io.load_street_graph(
    process_path + "street_graph_edges.gpkg", process_path + "street_graph_nodes.gpkg", crs=CRS_internal
)

print("Load perimeters")
perimeters_gdf = snman.io.load_perimeters(inputs_path + "perimeters/perimeters.shp", crs=CRS_internal)

print("Load rebuilding regions")
# Polygons that define which streets will be reorganized
rebuilding_regions_gdf = snman.io.load_rebuilding_regions(
    inputs_path + "rebuilding_regions/rebuilding_regions.gpkg", crs=CRS_internal
)

print("Load measurement regions")
# Polygons that define areas where network measures will be calculated
measurement_regions_gdf = snman.io.load_measurement_regions(
    inputs_path + "measurement_regions/measurement_regions.gpkg", crs=CRS_internal
)

print("Rebuild regions")
snman.rebuilding.rebuild_regions(  # TODO: multi_rebuild_regions --> doesn't work because no output L
    G,
    rebuilding_regions_gdf,
    rebuilding_function=lane_optimization,
    verbose=True,
    export_L=(export_path + "L_edges.gpkg", export_path + "L_nodes.gpkg"),
    export_H=(export_path + "H_edges.gpkg", export_path + "H_nodes.gpkg"),
)
