import pandas as pd
import geopandas as gpd
import numpy as np
from shapely.geometry import LineString


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


def extend_od_circular(od, nodes):
    """
    Create new OD matrix that ensures connectivity by connecting one node to the next in a list
    od: pd.DataFrame, original OD with columns s, t and trips_per_day
    nodes: list, all nodes in the graph
    """
    # shuffle and convert to df
    new_od_paths = pd.DataFrame(nodes, columns=["s"]).sample(frac=1).reset_index(drop=True)
    # shift by one to ensure cirularity
    new_od_paths["t"] = new_od_paths["s"].shift(1)
    # fille nan
    new_od_paths.loc[0, "t"] = new_od_paths.iloc[-1]["s"]

    # concatenate and add flow of 0 to the new OD pairs
    od_new = pd.concat([od, new_od_paths]).fillna(0).astype(int)

    # make sure that no loops
    od_new = od_new[od_new["s"] != od_new["t"]]
    return od_new.drop_duplicates(subset=["s", "t"])


def extend_od_matrix(od, nodes):
    """
    Extend the OD matrix such that every node appears as s and every node appears as t
    od: initial OD matrix, represented as a pd.Dataframe
    nodes: list of nodes in the graph
    """

    def get_missing_nodes(od_df):
        nodes_not_in_s = [n for n in nodes if n not in od_df["s"].values]
        nodes_not_in_t = [n for n in nodes if n not in od_df["t"].values]
        return nodes_not_in_s, nodes_not_in_t

    # find missing nodes
    nodes_not_in_s, nodes_not_in_t = get_missing_nodes(od)
    min_len = min([len(nodes_not_in_s), len(nodes_not_in_t)])
    len_diff = max([len(nodes_not_in_s), len(nodes_not_in_t)]) - min_len
    # combine every node of the longer list with a random permutation of the smaller list and add up
    if min_len == len(nodes_not_in_t):
        shuffled_t = np.random.permutation(nodes_not_in_t)
        combined_nodes = np.concatenate(
            [
                np.stack([nodes_not_in_s[:min_len], shuffled_t]),
                np.stack([nodes_not_in_s[min_len:], shuffled_t[:len_diff]]),
            ],
            axis=1,
        )
    else:
        shuffled_s = np.random.permutation(nodes_not_in_s)
        combined_nodes = np.concatenate(
            [
                np.stack([shuffled_s, nodes_not_in_t[:min_len]]),
                np.stack([shuffled_s[:len_diff], nodes_not_in_t[min_len:]]),
            ],
            axis=1,
        )
    # transform to dataframe
    new_od_paths = pd.DataFrame(combined_nodes.swapaxes(1, 0), columns=["s", "t"])
    # concat and add a flow value of 0 since we don't want to optimize the travel time for these lanes
    od_new = pd.concat([od, new_od_paths]).fillna(0)

    # check again
    nodes_not_in_s, nodes_not_in_t = get_missing_nodes(od_new)
    assert len(nodes_not_in_s) == 0 and len(nodes_not_in_t) == 0
    return od_new


def match_od_with_nodes(station_data_path: str, nodes: gpd.GeoDataFrame):
    """
    Match a OD matrix of coordinates with the node IDs

    Args:
        station_data_path (str): path to folder for one city (district)
        nodes (gpd.GeoDataFrame): List of graph nodes with geometry
    Returns:
        pd.DataFrame with columns s, t and trips, containint origin and destination node id and the number of trips
    """

    def linestring_from_coords(row):
        return LineString([[row["start_lng"], row["start_lat"]], [row["end_lng"], row["end_lat"]]])

    station_data = pd.read_csv(station_data_path)
    # print("Whole city OD matrix", len(station_data))

    # create linestring and convert to geodataframe
    station_data["geometry"] = station_data.apply(linestring_from_coords, axis=1)
    station_data = gpd.GeoDataFrame(station_data)
    station_data.set_geometry("geometry", inplace=True)

    if "birchplatz" in station_data_path or "affoltern" in station_data_path or "zurich" in station_data_path:
        original_crs = "EPSG:2056"
        station_data.crs = original_crs
        nodes.to_crs(2056, inplace=True)
    else:
        original_crs = "EPSG:4326"
        station_data.crs = original_crs
        if "cambridge" in station_data_path:
            nodes.to_crs("EPSG:2249", inplace=True)
            station_data.to_crs("EPSG:2249", inplace=True)
        elif "chicago" in station_data_path:
            nodes.to_crs("EPSG:26971", inplace=True)
            station_data.to_crs("EPSG:26971", inplace=True)
        else:
            raise NotImplementedError("Unknown city")
        station_data = station_data[station_data.geometry.is_valid]

    # select only the rows where the linestring intersects the area polygon
    area_polygon = gpd.GeoDataFrame(geometry=[nodes.geometry.unary_union.convex_hull], crs=nodes.crs)
    trips = station_data.sjoin(area_polygon)

    # get the closest nodes to the respective destination
    trips["geom_destination"] = gpd.points_from_xy(x=trips["end_lng"], y=trips["end_lat"])
    trips.set_geometry("geom_destination", inplace=True, crs=original_crs)
    trips.to_crs(nodes.crs, inplace=True)
    trips = trips.sjoin_nearest(nodes, distance_col="dist_destination", how="left", lsuffix="", rsuffix="destination")
    trips.rename(columns={"osmid": "osmid_destination"}, inplace=True)

    # set geometry to origin
    trips["geom_origin"] = gpd.points_from_xy(x=trips["start_lng"], y=trips["start_lat"])
    # trips["geom_origin"].apply(wkt.loads)
    trips.set_geometry("geom_origin", inplace=True, crs=original_crs)
    trips.to_crs(nodes.crs, inplace=True)
    trips.drop(["geom_destination"], axis=1, inplace=True)

    # get the closest nodes to the respective origin
    trips = trips.sjoin_nearest(nodes, distance_col="dist_origin", how="left", lsuffix="", rsuffix="origin")
    trips.rename(columns={"osmid": "osmid_origin"}, inplace=True)
    trips_final = (
        trips.groupby(["osmid_origin", "osmid_destination"])
        .agg({"count": "sum"})
        .reset_index()
        .rename(columns={"osmid_origin": "s", "osmid_destination": "t", "count": "trips"})
    )
    print("Number of OD-pairs (nodes):", len(trips_final), "Number of trips:", trips_final["trips"].sum())
    return trips_final
