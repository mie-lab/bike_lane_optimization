import os
import json
import numpy as np
import pandas as pd
import geopandas as gpd

from sqlalchemy import create_engine
import psycopg2

# configuratio & pathas
PATH_DATA = "../street_network_data/zurich/"
DB_LOGIN_PATH = "../../dblogin_mielab.json"
SCHEMA = "webapp"
CRS = 2056

# load the whole-city trip and construct origin and destination geometry
trips_microcensus = gpd.read_file(os.path.join(PATH_DATA, "raw_od_matrix", "trips_mc_cleaned_proj.gpkg"))
trips_microcensus["geom_destination"] = gpd.points_from_xy(
    x=trips_microcensus["end_lng"], y=trips_microcensus["end_lat"]
)
trips_microcensus["geom_origin"] = gpd.points_from_xy(
    x=trips_microcensus["start_lng"], y=trips_microcensus["start_lat"]
)

# load prebuilt OD matrix
od_whole_zurich_nodes = pd.read_csv(os.path.join(PATH_DATA, "od_matrix.csv"))


# Setup database access
def get_database_connector(dblogin_file=DB_LOGIN_PATH):
    """Create database connection with login json file"""
    with open(dblogin_file, "r") as infile:
        db_login = json.load(infile)
        db_login["database"] = "ebikecity"

    def get_con_mie():
        return psycopg2.connect(**db_login)

    return create_engine("postgresql+psycopg2://", creator=get_con_mie)


def get_expected_time_linear(nr_variables, coef=2.17235844e-05, intercept=-15.15954725242839):
    """Fit linear function to runtime from number of variables"""
    return nr_variables * coef + intercept


def get_expected_time(nr_variables):
    """Computes expected runtime (in min) from number of variables"""
    return 0.8 * np.exp(nr_variables / 1000000 * 1.6)


def compute_nr_variables(nr_edges, od_len):
    """Compute number of variables"""
    nr_variables = 3 * (od_len * nr_edges) + 2 * nr_edges
    return nr_variables


def generate_od_nodes(nodes):
    assert nodes.index.name == "osmid"
    nodes_in_area = nodes.index.unique()
    od_in_area = od_whole_zurich_nodes[
        (od_whole_zurich_nodes["s"].isin(nodes_in_area)) & (od_whole_zurich_nodes["t"].isin(nodes_in_area))
    ]
    return od_in_area


def generate_od_geometry(area_polygon, nodes):
    # restrict trips to the ones crossing the area
    trips = trips_microcensus.sjoin(area_polygon)

    # get the closest nodes to the respective destination
    trips.set_geometry("geom_destination", inplace=True, crs=CRS)
    trips.to_crs(nodes.crs, inplace=True)
    trips = trips.sjoin_nearest(nodes, distance_col="dist_destination", how="left", lsuffix="", rsuffix="destination")
    trips.rename(columns={"osmid": "osmid_destination"}, inplace=True)

    # trips["geom_origin"].apply(wkt.loads)
    trips.set_geometry("geom_origin", inplace=True, crs=CRS)
    trips.to_crs(nodes.crs, inplace=True)
    trips.drop(["geom_destination"], axis=1, inplace=True)

    # get the closest nodes to the respective origin
    trips = trips.sjoin_nearest(nodes, distance_col="dist_origin", how="left", lsuffix="", rsuffix="origin")
    trips.rename(columns={"osmid": "osmid_origin"}, inplace=True)
    od_in_area = (
        trips.groupby(["osmid_origin", "osmid_destination"])
        .agg({"count": "sum"})
        .reset_index()
        .rename(columns={"osmid_origin": "s", "osmid_destination": "t", "count": "trips"})
    )
    return od_in_area
