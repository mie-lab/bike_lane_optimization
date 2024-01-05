import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import geopandas as gpd
import os
from shapely import wkt

"""
NOTE: This preprocessing is based on code from two other repositories, namely the synpp pipeline and the V2G4Carsharing project.
The synpp pipeline generates inputs to the MATSim simulator in the form of activity scedules. The code is here: https://github.com/eqasim-org/synpp
In th context of the V2G4Carsharing project, we have developed code to transform the activity schedules into trips. 
The code can be found here: https://github.com/mie-lab/v2g4carsharing 
"""
from v2g4carsharing.trips_preparation.simulated_data_preprocessing import SimTripProcessor

# cache path --> containing outputs from synpp pipeline
cache_path = "synpp/cache/"
# output path
path = "street_network_data/od_matrix"
DISTANCE_THRESH = 5000  # general threshold
DIST_CUTOFF = 1000  # final threshold how to restrict the trips

os.makedirs(path, exist_ok=True)

processor = SimTripProcessor(cache_path, path)
# transform subsequent activities into trips
processor.transform_to_trips()
processor.save_trips()

trips = pd.read_csv(os.path.join(path, "trips_enriched.csv"))

trips["geom_destination"] = trips["geom_destination"].apply(wkt.loads)
trips = gpd.GeoDataFrame(trips, geometry="geom_destination")

# load nodes
nodes = gpd.read_file("street_network_data/nodes.gpkg")

# merge by the destination
len_trips_before = len(trips)
trips = trips.sjoin_nearest(nodes, distance_col="dist_destination", how="left", lsuffix="", rsuffix="destination")
assert len(trips) == len_trips_before

trips.rename(columns={"osmid": "osmid_destination"}, inplace=True)

# delete the ones that are not in the region
print(len(trips))
trips = trips[trips["dist_destination"] < DISTANCE_THRESH]
len(trips)

trips["geom_origin"] = trips["geom_origin"].apply(wkt.loads)
trips.drop(["geom_destination"], axis=1, inplace=True)
trips.set_geometry("geom_origin", inplace=True)

trips = trips.sjoin_nearest(nodes, distance_col="dist_origin", how="left", lsuffix="", rsuffix="origin")
len(trips)
trips.rename(columns={"osmid": "osmid_origin"}, inplace=True)

nearest_nodes = trips[
    ["person_id", "start_time_sec_destination", "osmid_origin", "dist_origin", "osmid_destination", "dist_destination"]
]

nearest_nodes.rename(
    columns={"index_origin": "node_id_origin", "index_destination": "node_id_destination"}, inplace=True
)

# code to search for a good distance cutoff
print("Fraction origin further away:", sum(nearest_nodes["dist_origin"] > DIST_CUTOFF) / len(nearest_nodes))
print("Fraction destination further away:", sum(nearest_nodes["dist_destination"] > DIST_CUTOFF) / len(nearest_nodes))

nearest_nodes = nearest_nodes[
    (nearest_nodes["dist_origin"] < DIST_CUTOFF) & (nearest_nodes["dist_destination"] < DIST_CUTOFF)
]

# nearest_nodes.to_csv(os.path.join(path, "trips_node_ids.csv"))

od_matrix = nearest_nodes.groupby(["osmid_origin", "osmid_destination"])["person_id"].count()
od_matrix = od_matrix.reset_index().rename({"person_id": "trips"}, axis=1)
od_matrix.to_csv(os.path.join(path, "od_matrix.csv"), index=False)
