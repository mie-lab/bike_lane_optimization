import argparse
import os
import numpy as np
import pandas as pd
import pyproj
import geopandas as gpd
from shapely.geometry import LineString

from ebike_city_tools.utils import match_od_with_nodes

CH1903 = "epsg:21781"
LV05 = CH1903
CH1903_PLUS = "epsg:2056"
LV95 = CH1903_PLUS

filename_mapping = {"chicago": "-divvy-tripdata.csv", "cambridge": "-bluebikes-tripdata.csv"}
column_name_mapping = {
    "end station latitude": "end_lat",
    "start station latitude": "start_lat",
    "end station longitude": "end_lng",
    "start station longitude": "start_lng",
    "start station id": "start_station_id",
    "end station id": "end_station_id",
}


def bike_sharing_preprocessing(data_path: str) -> None:
    """
    Preprocess raw od data from bike sharing company for Chicago

    Args:
        data_path (str): path to data folder
    """
    all_data = []
    for i in np.arange(1, 13):
        print("processing file", i)
        data = pd.read_csv(
            os.path.join(
                data_path, "raw_od_matrix", f"2023{str(i).zfill(2)}" + filename_mapping[data_path.split(os.sep)[-1]]
            )
        )
        data.rename(columns=column_name_mapping, inplace=True)
        data.dropna(subset=["start_station_id", "end_station_id"], inplace=True)
        data["start_station_id"] = data["start_station_id"].astype(str)
        data["end_station_id"] = data["end_station_id"].astype(str)
        data = data[
            (~data["start_station_id"].str.contains("checking"))
            & (~data["start_station_id"].str.contains("charg"))
            & (~data["end_station_id"].str.contains("checking"))
            & (~data["end_station_id"].str.contains("charg"))
        ]
        station_data = (
            data.groupby(["start_lat", "start_lng", "end_lat", "end_lng"])
            .agg(
                {"start_lat": "count"}
                #     # first version: group by station ids --> it's slightly less than using the stations
                #     station_data = data.groupby(["start_station_id", "end_station_id"]).agg(
                # {"ride_id": "count", "start_lat": "mean", "start_lng": "mean", "end_lat": "mean", "end_lng": "mean"}
            )
            .rename(columns={"start_lat": "count"})
            .reset_index()
        )
        all_data.append(station_data)
    all_data = pd.concat(all_data)
    # group by stations again
    all_data = all_data.groupby(["start_lat", "start_lng", "end_lat", "end_lng"]).agg({"count": "sum"}).reset_index()
    all_data.to_csv(os.path.join(data_path, "raw_od_matrix", "od_whole_city.csv"), index=False)


def zurich_preprocessing(data_path: str) -> None:
    """
    Preprocess raw od data from Mobility Microcensus in Zurich

    Args:
        data_path (str): path to zurich data folder
    """
    df_mz_trips = pd.read_csv(os.path.join(data_path, "raw_od_matrix/wege.csv"), encoding="latin-1")
    df_mz_trips.loc[:, "trip_id"] = df_mz_trips["WEGNR"]

    # Adjust coordinates
    for mz_attribute, df_attribute in [("Z", "destination"), ("S", "origin"), ("W", "home")]:
        coords = df_mz_trips[["%s_X_CH1903" % mz_attribute, "%s_Y_CH1903" % mz_attribute]].values
        transformer = pyproj.Transformer.from_crs(CH1903, CH1903_PLUS)
        x, y = transformer.transform(coords[:, 0], coords[:, 1])
        df_mz_trips.loc[:, "%s_x" % df_attribute] = x
        df_mz_trips.loc[:, "%s_y" % df_attribute] = y

    # select relevant columns
    station_data = df_mz_trips[["destination_x", "destination_y", "origin_x", "origin_y"]]
    station_data.rename(
        {"destination_x": "end_lng", "destination_y": "end_lat", "origin_x": "start_lng", "origin_y": "start_lat"},
        axis=1,
        inplace=True,
    )
    station_data["count"] = 1
    station_data.to_csv(os.path.join(data_path, "raw_od_matrix", "od_whole_city.csv"), index=False)


def match_od_with_nodes_path(data_path: str) -> pd.DataFrame:
    """
    Match a row OD matrix of coordinates with the node IDs

    Args:
        data_path (str): path to folder for one city (district)
    Returns:
        pd.DataFrame with columns s, t and trips, containint origin and destination node id and the number of trips
    """
    nodes = gpd.read_file(os.path.join(data_path, "nodes_all_attributes.gpkg"))
    station_data_path = os.path.join(data_path, "raw_od_matrix", "od_whole_city.csv")
    match_od_with_nodes(station_data_path, nodes)


def reduce_od_by_trip_ratio(od: pd.DataFrame, trip_ratio: float = 0.75) -> pd.DataFrame:
    """
    Reduce OD matrix by including only the x trips that collectively make up for <trip_ratio> of the trips

    Args:
        od (pd.DataFrame): OD matrix with columns s, t, trips
        trip_ratio (float, optional): Ratio of trips that the OD paths need to make up. Defaults to 0.75.

    Returns:
        pd.DataFrame: reduced OD matrix
    """
    # Sort the DataFrame by visits in descending order
    df_sorted = od.sort_values(by="trips", ascending=False)

    # Calculate the cumulative sum of visits
    df_sorted["cumulative_trips"] = df_sorted["trips"].cumsum()

    # Calculate the total sum of visits
    total_visits = df_sorted["trips"].sum()

    # Find the s-t pairs where the cumulative sum of trips is just over trip_ratio
    threshold = total_visits * trip_ratio
    most_frequent_trips = df_sorted[df_sorted["cumulative_trips"] <= threshold]
    return most_frequent_trips.drop(["cumulative_trips"], axis=1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data_path", default="../street_network_data/affoltern", type=str)
    parser.add_argument("-r", "--trip_ratio", default=0.75, type=float)
    args = parser.parse_args()

    data_path = args.data_path
    # preprocessing of bike sharing data
    # bike_sharing_preprocessing(data_path)

    # create od matrix based on the graph nodes
    od_matrix = match_od_with_nodes_path(data_path)

    # for chicago and cambridge: reduce od matrix size to 75% most frequent trips
    if "chicago" in data_path or "cambridge" in data_path:
        od_matrix = reduce_od_by_trip_ratio(od_matrix, args.trip_ratio)

    print("FINAL OD pairs", len(od_matrix))

    od_matrix.to_csv(os.path.join(data_path, "od_matrix.csv"), index=False)
