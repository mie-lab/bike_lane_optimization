import pandas as pd
import scipy
from pandas import DataFrame
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import geopandas as gpd
from typing import Union, Any
from sqlalchemy.types import Text
from scipy.stats import entropy
from shapely import LineString, Point
from numpy.typing import NDArray
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import norm as sparse_norm
import numpy as np
import scipy.linalg


def get_expected_runtime(edges_df: gpd.GeoDataFrame, eval_metric: str) -> float:

    num_rows = len(edges_df)
    scaling_factors = {
        'bci': 0.0026,
        'lts': 0.0008,
        'blos': 0.0013,
        'bls': 0.0020,
        'porter': 0.0152,
        'weikl': 0.0168
    }

    if eval_metric not in scaling_factors:
        raise ValueError(f"Unknown evaluation metric: {eval_metric}")

    scaling_factor = scaling_factors[eval_metric]
    expected_runtime = scaling_factor * num_rows

    return expected_runtime


def calculate_lts(
        edges_df: gpd.GeoDataFrame,
        lane_col: str,
        speed_limits: gpd.GeoDataFrame,
        speed_col: str,
        traffic_volume: gpd.GeoDataFrame,
        traffic_col: str,
        output_col: str = 'lts',
        buffer: float = 8,
        invert: bool = False
) -> gpd.GeoDataFrame:
    """
    Calculate the Level of Traffic Stress (LTS) for a road network (Mercuria et al., 2012).

    :param edges_df: GeoDataFrame containing the road network edges.
    :param lane_col: Column in `edges_df` representing lane configuration data.
    :param speed_limits: GeoDataFrame containing speed limit data for roads.
    :param speed_col: Column name representing speed limits.
    :param traffic_volume: GeoDataFrame containing traffic volume data.
    :param traffic_col: Column name  representing traffic volumes.
    :param output_col: The column name in the output GeoDataFrame where the LTS score will be stored.
    :param buffer: Default buffer distance (in meters) for spatial operations.
    :param invert: Whether to invert the resulting LTS values (e.g., reverse the scale).
    :return: GeoDataFrame with the calculated LTS score added as a new column.
    """

    df = edges_df.copy()
    df['car_lane_n'] = df[lane_col].apply(lambda x: count_characters(x, "HMT")).replace(0, np.nan)
    df['bike_lane_bool'] = df[lane_col].apply(lambda x: 1 if pd.notna(x) and any(char in "P" for char in x) else 0)
    df['side_parking_bool'] = df[lane_col].apply(lambda x: count_characters(x, "RND") > 0)
    df['oneway_bool'] = df[lane_col].apply(lambda x: 1 if pd.notna(x) and all(char in x for char in "<>") else 0)

    buff_edges = calculate_buffer(df.copy(), buffer)
    _ = buff_edges.sindex

    df = merge_spatial_attribute(buff_edges, df, speed_limits, speed_col)
    df = merge_spatial_attribute(buff_edges, df, traffic_volume, traffic_col)

    lts_grades = []

    for edge in df.index:
        car_lanes = df.loc[edge, 'car_lane_n']
        has_bike_lane = df.loc[edge, 'bike_lane_bool']
        has_side_parking = df.loc[edge, 'side_parking_bool']
        oneway = df.loc[edge, 'oneway_bool']
        traffic_volume = df.loc[edge, traffic_col]
        speed_limit = df.loc[edge, speed_col]
        cur_lts = 3

        if speed_limit <= 20:
            cur_lts = 1
        elif has_bike_lane:
            if speed_limit == 30:
                if car_lanes >= 2 and not oneway:
                    cur_lts = 3
                else:
                    cur_lts = 2
            elif speed_limit >= 50:
                cur_lts = 3
        else:
            if not has_side_parking:
                if speed_limit == 30:
                    if traffic_volume < 1000:
                        cur_lts = 1
                    elif 1000 <= traffic_volume <= 3000:
                        cur_lts = 2
                    elif car_lanes == 1 or oneway:
                        cur_lts = 2
                    else:
                        cur_lts = 3
                elif speed_limit >= 50:
                    if traffic_volume <= 3000:
                        cur_lts = 3
                    elif traffic_volume > 3000:
                        cur_lts = 4
            elif has_side_parking:
                if speed_limit == 30:
                    if traffic_volume < 1000:
                        cur_lts = 2
                elif speed_limit >= 50:
                    if traffic_volume > 3000:
                        cur_lts = 4

        lts_grades.append(cur_lts)

    if invert:
        lts_grades = [5 - x for x in lts_grades]

    df[output_col] = lts_grades

    return df


def calculate_bci(
        edge_df: gpd.GeoDataFrame,
        lane_col: str,
        bike_lane_width_col: str,
        motorized_width_col: str,
        landuse: gpd.GeoDataFrame,
        landuse_col: str,
        traffic_volume: gpd.GeoDataFrame,
        traffic_col: str,
        speed_limits: gpd.GeoDataFrame,
        speed_col: str,
        output_col: str = 'bci',
        buffer: float = 8,
        invert: bool = False
) -> gpd.GeoDataFrame:
    """
    Calculate the Bicycle Compatibility Index (BCI) for a road network (Harvey, 1998).

    :param edge_df: GeoDataFrame containing the road network edges.
    :param lane_col: Column in `edge_df` representing the lane configuration data.
    :param bike_lane_width_col: Column in `edge_df` representing the width of bike lanes.
    :param motorized_width_col: Column in `edge_df` representing the width of motorized road section.
    :param landuse: GeoDataFrame containing land use data (e.g., residential areas).
    :param landuse_col: Column name representing land use type.
    :param traffic_volume: GeoDataFrame containing traffic volume data.
    :param traffic_col: Column name representing traffic volumes.
    :param speed_limits: GeoDataFrame containing speed limit data for roads.
    :param speed_col: Column name representing speed limits.
    :param output_col: The column name in the output GeoDataFrame where the BCI score will be stored.
    :param buffer: Buffer distance (in meters) for spatial operations.
    :param invert: Whether to invert the resulting BCI values (e.g., reverse the scale).
    :return: GeoDataFrame with the calculated BCI score and associated grades added as new columns.
    """

    df = edge_df.copy()
    buff_edges = calculate_buffer(df, buffer)
    df = merge_spatial_attribute(buff_edges, df, speed_limits, speed_col)
    df = merge_spatial_attribute(buff_edges, df, traffic_volume, traffic_col)
    df = merge_spatial_boolean(buff_edges, df, landuse[landuse[landuse_col] == 'residential'],
                               target_col='res_landuse_bool', threshold_col='length', threshold=75)

    df['bike_lane_bool'] = df[lane_col].apply(lambda x: 1 if pd.notna(x) and any(char in "P" for char in x) else 0)
    df['car_lane_n'] = df[lane_col].apply(lambda x: count_characters(x, "HMT")).replace(0, np.nan)
    df['side_parking_bool'] = df[lane_col].apply(lambda x: count_characters(x, "RND") > 0)
    df['bike_lane_width'] = np.where((df[bike_lane_width_col] == 0) & (df['bike_lane_bool'] == 1), 1.5,
                                     df[bike_lane_width_col])
    df['curb_lane_width'] = df[motorized_width_col] / df['car_lane_n']
    df['curb_lane_vol'] = df[traffic_col] / df['car_lane_n']

    df[output_col] = (
            3.67
            - 0.966 * df['bike_lane_bool']
            - 0.410 * df['bike_lane_width']
            - 0.498 * df['curb_lane_width']
            + 0.002 * df['curb_lane_vol']
            + 0.0004 * (df[traffic_col] - df['curb_lane_vol'])
            + 0.022 * df[speed_col]
            + 0.506 * df['side_parking_bool']
            - 0.264 * df['res_landuse_bool']
    )

    percentiles = df[output_col].quantile([0.05, 0.25, 0.5, 0.75, 0.95])
    bins = [df[output_col].min(), percentiles[0.05], percentiles[0.25], percentiles[0.5], percentiles[0.75],
            percentiles[0.95], df[output_col].max()]
    grades = ['A', 'B', 'C', 'D', 'E', 'F']

    # Assign grades
    df[f'{output_col}_grade'] = pd.cut(df[output_col], bins=bins, labels=grades, include_lowest=True)

    if invert:
        bci_max = df[output_col].max()
        df[output_col] = bci_max - df[output_col]

    return df


def calculate_bsl(
        edges_df: gpd.GeoDataFrame,
        lane_col: str,
        motorized_width_col: str,
        speed_limits: gpd.GeoDataFrame,
        speed_col: str,
        traffic_volume: gpd.GeoDataFrame,
        traffic_col: str,
        output_col: str = 'bsl',
        buffer: float = 8,
        invert: bool = False
) -> gpd.GeoDataFrame:
    """
    Calculate the Bicycle Stress Level (BSL) for a road network.

    :param edges_df: GeoDataFrame containing the road network edges.
    :param lane_col: Column in `edges_df` representing the lane configuration data.
    :param motorized_width_col: Column in `edges_df` representing the width of motorized lanes or road sections.
    :param speed_limits: GeoDataFrame containing speed limit data for roads.
    :param speed_col: Column name representing speed limits.
    :param traffic_volume: GeoDataFrame containing traffic volume data.
    :param traffic_col: Column name representing traffic volumes.
    :param output_col: The column name in the output GeoDataFrame where the BSL score will be stored.
    :param buffer: Default buffer distance (in meters) for spatial operations.
    :param invert: Whether to invert the resulting BSL values (e.g., reverse the scale).
    :return: GeoDataFrame with the calculated BSL score added as a new column.
    """

    df = edges_df.copy()
    buff_edges = calculate_buffer(df, buffer)

    df = merge_spatial_attribute(buff_edges, df, speed_limits, speed_col)
    df = merge_spatial_attribute(buff_edges, df, traffic_volume, traffic_col)
    df['car_lane_n'] = df[lane_col].apply(lambda x: count_characters(x, "HMT")).replace(0, np.nan)
    df['curb_lane_width'] = df[motorized_width_col] / df['car_lane_n']
    df['curb_lane_vol'] = df[traffic_col] / df['car_lane_n']

    points = [1, 2, 3, 4, 5]
    conditions = [
        df[speed_col] >= 75,
        df[speed_col] >= 65,
        df[speed_col] >= 60,
        df[speed_col] >= 50,
        df[speed_col] <= 40
    ]
    df['speed_points'] = np.select(conditions, points, default=np.nan)

    conditions_width = [
        df['curb_lane_width'] >= 4.6,
        (df['curb_lane_width'] < 4.6) & (df['curb_lane_width'] >= 4.3),
        (df['curb_lane_width'] < 4.3) & (df['curb_lane_width'] >= 4.0),
        (df['curb_lane_width'] < 4.0) & (df['curb_lane_width'] >= 3.7),
        df['curb_lane_width'] <= 3.3
    ]
    df['curb_width_points'] = np.select(conditions_width, points, default=np.nan)

    conditions_volume = [
        df['curb_lane_vol'] >= 450,
        df['curb_lane_vol'] <= 350,
        df['curb_lane_vol'] <= 250,
        df['curb_lane_vol'] <= 150,
        df['curb_lane_vol'] <= 50
    ]
    df['curb_vol_points'] = np.select(conditions_volume, points, default=np.nan)
    df[output_col] = df[['curb_vol_points', 'curb_width_points', 'speed_points']].mean(axis=1)

    if invert:
        df[output_col] = 6 - df[output_col]

    return df


def calculate_blos(
        edges_df: gpd.GeoDataFrame,
        lane_col: str,
        traffic_volume: gpd.GeoDataFrame,
        traffic_cols: list,
        motorized_width_col: str,
        speed_limits: gpd.GeoDataFrame,
        speed_col: str,
        surface: gpd.GeoDataFrame,
        surface_col: str,
        output_column:str = 'blos',
        buffer: float = 8,
        invert: bool = False
) -> gpd.GeoDataFrame:
    """
    Calculate the Bicycle Level of Service (BLOS) for a road network. Based on US Highway Manual 2005 edition.

    :param edges_df: GeoDataFrame containing the road network edges.
    :param lane_col: Column in `edges_df` representing lane data.
    :param speed_limits: GeoDataFrame containing speed limit data for roads.
    :param speed_col: Column name representing speed limits.
    :param traffic_volume: GeoDataFrame containing traffic volume data.
    :param traffic_cols: Column names representing traffic volumes by traffic type.
    :param motorized_width_col: Column in 'edges_df' representing motorized road area width.
    :param surface: GeoDataFrame containing surface quality data.
    :param surface_col: Column name representing surface conditions.
    :param output_column: The column name in the output GeoDataFrame where the BLOS will be stored.
    :param buffer: Default buffer distance (in meters) for spatial operations.
    :param invert: Whether to invert the resulting BLOS values.
    :return: GeoDataFrame with the calculated BLOS added as a new column.
    """

    df = edges_df.copy()
    buff_edges = calculate_buffer(df, buffer)
    df = merge_spatial_attribute(buff_edges, df, traffic_volume, traffic_cols)
    df = merge_spatial_attribute(buff_edges, df, speed_limits, speed_col)
    df = merge_spatial_attribute(buff_edges, df, surface, surface_col).replace(0, np.nan)
    df['oneway_bool'] = df[lane_col].apply(lambda x: pd.notna(x) and all(char in x for char in "<>"))
    df['car_lane_n'] = df[lane_col].apply(lambda x: count_characters(x, "HMT")).replace(0, np.nan)
    df['car_lane_n'] = np.where(df['oneway_bool'], df['car_lane_n'], df['car_lane_n'] / 2)
    df['curb_lane_width'] = df[motorized_width_col] / df['car_lane_n']
    df['heavy_veh'] = df[traffic_cols].sum(axis=1)

    traffic_vol_15_min = (df[traffic_cols[0]] * 0.565 * 0.1) / 4
    speed_limit_mph = df[speed_col] * 0.621371  # Convert kph to mph
    effective_speed_limit = np.where(speed_limit_mph > 20,
                                     1.1199 * np.log(np.maximum(speed_limit_mph - 20, 1e-10)) + 0.8103, 0)
    heavy_vehicles_percent = np.where(traffic_vol_15_min > 0, (df['heavy_veh'] / traffic_vol_15_min) * 100, 0)

    term1 = np.where((traffic_vol_15_min > 0) & (df['car_lane_n'] > 0),
                     0.507 * np.log(np.maximum(traffic_vol_15_min / df['car_lane_n'], 1e-10)), 0)
    term2 = 0.199 * effective_speed_limit * (1 + 10.38 * heavy_vehicles_percent) ** 2
    term3 = 7.066 * (1 / df[surface_col]) ** 2
    term4 = -0.005 * (df['curb_lane_width'] * 3.28084) ** 2  # Convert width to feet
    df[output_column] = term1 + term2 + term3 + term4 + 0.760

    bins = [-np.inf, 1.5, 2.5, 3.5, 4.5, 5.5, np.inf]
    grades = ['A', 'B', 'C', 'D', 'E', 'F']
    df[f'{output_column}_grade'] = pd.cut(df[output_column], bins=bins, labels=grades, include_lowest=True)

    if invert:
        df[output_column] = df[output_column].max() - df[output_column]

    return df


def calculate_porter_index(
        edges_df: gpd.GeoDataFrame,
        lane_col: str,
        housing: gpd.GeoDataFrame,
        population: gpd.GeoDataFrame,
        population_col: str,
        green_spaces: gpd.GeoDataFrame,
        trees: gpd.GeoDataFrame,
        pt_stops: gpd.GeoDataFrame,
        air_quality: gpd.GeoDataFrame,
        air_col: str,
        output_col: str = 'porter',
        invert: bool = False
) -> gpd.GeoDataFrame:
    """
    Calculate the Porter Index, which assesses the quality of urban bike lane networks (Porter et al., 2020).

    :param edges_df: GeoDataFrame containing the road or bike lane network edges.
    :param lane_col: Column in `edges_df` representing bike lane data.
    :param housing: GeoDataFrame containing data on housing units.
    :param population: GeoDataFrame containing data on population distribution.
    :param population_col: Column name representing the population count.
    :param green_spaces: GeoDataFrame containing data on green spaces or parks.
    :param trees: GeoDataFrame containing data on tree coverage areas.
    :param pt_stops: GeoDataFrame containing public transport stops.
    :param air_quality: GeoDataFrame containing air quality data.
    :param air_col: Column name representing air pollution levels.
    :param output_col: The column name in the output GeoDataFrame where the Porter Index score will be stored.
    :param invert: Whether to invert the resulting Porter Index values (e.g., reverse the scale).
    :return: GeoDataFrame with the calculated Porter Index score added as a new column.
    """

    df = edges_df.copy()
    buffer_30 = calculate_buffer(df, 30)
    buffer_15 = calculate_buffer(df, 15)

    bike_lanes = df[df[lane_col].str.contains("P", na=False)][['geometry', 'length']]
    df['bike_lane_density'] = calculate_count(buffer_30, bike_lanes, 'index', 'length')
    df['parks_n'] = calculate_count(buffer_30, green_spaces, 'index')
    df['housing_units'] = calculate_count(buffer_30, housing, 'index') / (buffer_30['buff_area'] / 1e6)
    df['pop_density'] = calculate_count(buffer_30, population, 'index', population_col) / (buffer_30['buff_area'] / 1e6)

    df = merge_spatial_share(df, buffer_15, trees, 'tree_coverage', 'buff_area', percent=True)
    df = merge_spatial_count(buffer_15, df, air_quality, target_col='air_poll', agg_col=air_col, agg_func=avg_no2)
    df = merge_distance_to_nearest(df, pt_stops, "dist_to_pt", merge_col='index', how='left')

    weights = {
        'bike_lane_density': 0.79,
        'housing_units': 0.87,
        'pop_density': 0.95,
        'air_poll': -0.62,
        'dist_to_pt': -0.46,
        'parks_n': 0.33,
        'tree_coverage': -0.44
    }

    scaler = MinMaxScaler()
    columns_to_normalize = list(weights.keys())
    df[columns_to_normalize] = scaler.fit_transform(df[columns_to_normalize])
    df[output_col] = sum(df[col] * weight for col, weight in weights.items())

    if invert:
        max_value = max(df[output_col])
        df[output_col] = [max_value - df[output_col]]

    return df


def calculate_weikl_index(
        edges_df: gpd.GeoDataFrame,
        lane_col: str,
        speed_limits: gpd.GeoDataFrame,
        speed_col: str,
        street_lighting: gpd.GeoDataFrame,
        traffic_volume: gpd.GeoDataFrame,
        traffic_col: str,
        slope: gpd.GeoDataFrame,
        slope_col: str,
        surface: gpd.GeoDataFrame,
        surface_col: str,
        green_spaces: gpd.GeoDataFrame,
        noise_pollution: gpd.GeoDataFrame,
        noise_col: str,
        air_quality: gpd.GeoDataFrame,
        agg_col: str,
        output_column: str = 'weikl',
        buffer: float = 8,
        bike_speed: float = 15,
        invert: bool = False
) -> gpd.GeoDataFrame:
    """
    Calculate the Weikl Index for urban bike lane quality (Weikl et al., 2023.

    :param edges_df: GeoDataFrame containing the road network edges.
    :param lane_col: Column in `edges_df` representing lane data.
    :param speed_limits: GeoDataFrame containing speed limit data for roads.
    :param speed_col: Column name representing speed limits.
    :param street_lighting: GeoDataFrame containing street lighting data.
    :param traffic_volume: GeoDataFrame containing traffic volume data.
    :param traffic_col: Column name representing traffic volumes.
    :param slope: GeoDataFrame containing slope data for roads.
    :param slope_col: Column name representing slope values.
    :param surface: GeoDataFrame containing surface quality data.
    :param surface_col: Column name representing surface conditions.
    :param green_spaces: GeoDataFrame containing green space data.
    :param noise_pollution: GeoDataFrame containing noise pollution levels for roads.
    :param noise_col: Column name representing noise levels.
    :param air_quality: GeoDataFrame containing air quality data.
    :param agg_col: Column name representing air quality levels.
    :param output_column: The column name in the output GeoDataFrame where the Weikl Index will be stored.
    :param buffer: Default buffer distance (in meters) for spatial operations.
    :param bike_speed: Default bike speed (in km/h).
    :param invert: Whether to invert the resulting Weikl Index values.
    :return: GeoDataFrame with the calculated Weikl Index added as a new column.
    """

    df = edges_df.copy()
    buffer = calculate_buffer(df.copy(), buffer)

    df['bike_lane_bool'] = df[lane_col].apply(lambda x: 1 if pd.notna(x) and any(char in "P" for char in x) else 0)
    df['speed_diff'] = merge_spatial_attribute(buffer, df, speed_limits, speed_col).fillna(0)[speed_col] - bike_speed
    df = merge_spatial_count(buffer, df, street_lighting, target_col='street_lights')
    df['light_ratio'] = df['street_lights'] / df['length']
    df = merge_spatial_share(df, buffer, green_spaces, 'green_space_share', 'buff_area')
    df = merge_spatial_count(buffer, df, air_quality, target_col='air_poll', agg_col=agg_col, agg_func=avg_no2)
    df = merge_spatial_attribute(buffer, df, traffic_volume, traffic_col)
    df = merge_spatial_attribute(buffer, df, slope, slope_col)
    df = merge_spatial_attribute(buffer, df, surface, surface_col).replace(0, np.nan)
    df = merge_spatial_attribute(buffer, df, noise_pollution, noise_col)

    # Assign points based on bins
    bins = {
        traffic_col: [0, 500, 2500, 5000],
        slope_col: [0.033, 0.075, 0.2, 0.4],
        noise_col: [60, 65, 70, 75],
        'air_poll': [20, 35, 50, 100],
        'green_space': [75, 50, 25, 0],
        'speed_diff': [35, 25, 15, 5],
        'light_ratio': [0.03, 0.02, 0.01, 0.005]
    }
    # Points
    bike_lane_pts = np.where(df['bike_lane_bool'] > 0, 5, 1)
    speed_pts = assign_points(df['speed_diff'], bins['speed_diff'])
    light_pts = assign_points(df['light_ratio'], bins['light_ratio'])
    slope_pts = assign_points(df[slope_col] ** 2 / df['length'], bins[slope_col])
    green_space_pts = assign_points(df['green_space_share'], bins['green_space'])
    traffic_vol_pts = assign_points(df[traffic_col], bins[traffic_col])
    noise_poll_pts = assign_points(df[noise_col], bins[noise_col])
    air_quality_pts = assign_points(df['air_poll'], bins['air_poll'])

    safety = 0.27 * bike_lane_pts + 0.11 * speed_pts + 0.23 * traffic_vol_pts + 0.13 * light_pts
    comfort = 0.27 * bike_lane_pts + 0.19 * slope_pts + 0.26 * df[surface_col].fillna(0)
    attr = 0.35 * green_space_pts + 0.3 * noise_poll_pts + 0.35 * air_quality_pts
    weikl_index = 0.3 * safety + 0.19 * comfort + 0.13 * attr

    if invert:
        weikl_index = weikl_index.max() - weikl_index

    df[output_column] = weikl_index

    return df


def assign_points(series, bins):
    """Assign points based on thresholds."""
    return np.digitize(series, bins, right=True).clip(1, 5)


def count_characters(value, chars):
    """Count char in lane composition."""
    if pd.isna(value):
        return 0  # Return 0 if the value is NaN
    return sum([1 for char in value if char in chars])


def calculate_buffer(
        df: gpd.GeoDataFrame,
        buffer_size: float
) -> gpd.GeoDataFrame:

    buff_df = df.copy()[['geometry', 'index']]
    buff_df['geometry'] = buff_df.buffer(buffer_size)
    buff_df['buff_area'] = buff_df.area

    return buff_df


def average_pop(group):
    """Calculate average population."""
    return group['PERS_N'].sum() / len(group)


def avg_no2(group):
    """Calculate average no2 emissions"""
    return group['no2'].sum() / len(group)


def calculate_count(
        df: gpd.GeoDataFrame,
        context_data: gpd.GeoDataFrame,
        grouping_column: str,
        count_column: str = None
) -> gpd.GeoDataFrame:
    """sums quantities of items or item properties."""

    overlaps = gpd.sjoin(df, context_data, how='inner', predicate='intersects')

    if count_column:
        result = overlaps.groupby(grouping_column)[count_column].sum()
    else:
        result = overlaps.groupby(grouping_column).size()

    return df[grouping_column].map(result).fillna(0)


def merge_spatial_attribute(
        buff_edges: gpd.GeoDataFrame,
        edges: gpd.GeoDataFrame,
        spatial_data: gpd.GeoDataFrame,
        attribute_cols: Union[str, list],
        target_cols: Union[str, list] = None,
        merge_col: str = "index"
) -> gpd.GeoDataFrame:

    # if only one column
    if not isinstance(attribute_cols, list):
        attribute_cols = [attribute_cols]

    if target_cols is None:
        target_cols = attribute_cols

    if not isinstance(target_cols, list):
        target_cols = [target_cols]

    # Perform spatial overlay
    overlaps = gpd.overlay(buff_edges, spatial_data, how="intersection", keep_geom_type=False)
    overlaps['overlap_length'] = overlaps.geometry.length
    idx = overlaps.groupby(merge_col)['overlap_length'].idxmax()
    max_overlaps = overlaps.loc[idx]

    merge_cols = [merge_col] + attribute_cols
    edges = edges.merge(max_overlaps[merge_cols], on=merge_col, how="left")
    rename_dict = dict(zip(attribute_cols, target_cols))
    edges = edges.rename(columns=rename_dict)

    return edges


def merge_spatial_boolean(
        buff_edges: gpd.GeoDataFrame,
        edges: gpd.GeoDataFrame,
        spatial_data: gpd.GeoDataFrame,
        target_col: str,
        threshold_col: str,
        merge_col: str = "index",
        threshold: float = 0
) -> gpd.GeoDataFrame:

    overlaps = gpd.overlay(buff_edges, spatial_data, how="intersection", keep_geom_type=False)
    overlaps['overlap_length'] = overlaps.geometry.length
    overlap_sums = overlaps.groupby(merge_col)['overlap_length'].sum().reset_index(name='overlap_length_sum')
    edges = edges.merge(overlap_sums, on=merge_col, how='left')

    # Fill NaN values in overlap_length_sum with 0 (no overlap)
    edges['overlap_length_sum'] = edges['overlap_length_sum'].fillna(0)
    edges[target_col] = (edges['overlap_length_sum'] / edges[threshold_col]) * 100 > threshold
    edges = edges.drop(columns=['overlap_length_sum'])

    return edges


def merge_distance_to_nearest(
        edges: gpd.GeoDataFrame,
        spatial_data: gpd.GeoDataFrame,
        target_col: str,
        merge_col: str = "index",
        how: str = "intersection"
) -> DataFrame:
    nearest_stops = gpd.sjoin_nearest(edges, spatial_data, how=how, distance_col=target_col)
    nearest_stops = nearest_stops.drop_duplicates(subset=merge_col)
    edges = edges.merge(nearest_stops[[merge_col, target_col]], on=merge_col)

    return edges


def merge_spatial_count(
        buff_edges: gpd.GeoDataFrame,
        edges: gpd.GeoDataFrame,
        spatial_data: gpd.GeoDataFrame,
        target_col: str,
        agg_col: str = None,
        agg_func: Any = 'size',
        merge_col: str = "index",
) -> gpd.GeoDataFrame:

    overlaps = gpd.overlay(buff_edges, spatial_data, how="intersection", keep_geom_type=False)

    if agg_func in ['size', 'sum', 'min', 'max']:
        aggregation = overlaps.groupby(merge_col)[agg_col].agg(agg_func) if agg_col else overlaps.groupby(
            merge_col).agg(agg_func)
    else:
        aggregation = overlaps.groupby(merge_col).apply(agg_func)

    aggregation_aligned = buff_edges[merge_col].map(aggregation).fillna(0)
    edges[target_col] = aggregation_aligned

    return edges


def merge_distance_to_nearest(
        edges: gpd.GeoDataFrame,
        spatial_data: gpd.GeoDataFrame,
        target_col: str,
        merge_col: str = "index",
        how: str = "intersection"
) -> gpd.GeoDataFrame:

    nearest_stops = gpd.sjoin_nearest(edges, spatial_data, how=how, distance_col=target_col)
    edges = edges.merge(nearest_stops[[merge_col, target_col]], on=merge_col)

    return edges


def merge_spatial_share(
        edges: gpd.GeoDataFrame,
        buffer: gpd.GeoDataFrame,
        spatial_data: gpd.GeoDataFrame,
        target_col: str,
        divider_col: str,
        percent: bool = False,
        merge_col: str = "index",
) -> gpd.GeoDataFrame:

    overlaps = gpd.overlay(buffer, spatial_data, how="intersection", keep_geom_type=False)
    print(overlaps.columns)

    if overlaps.empty:
        edges[target_col] = 0
        return edges

    if edges.geometry.type.iloc[0] == 'LineString':
        overlaps['overlap'] = overlaps.geometry.length
    elif edges.geometry.type.iloc[0] == 'Polygon':
        overlaps['overlap'] = overlaps.geometry.area

    overlap_sums = overlaps.groupby(merge_col)['overlap'].sum().rename('overlap_sum')
    edges = edges.merge(overlap_sums, on=merge_col, how='left')

    edges[target_col] = (edges['overlap_sum'] / buffer[divider_col]).fillna(0)
    if percent:
        edges[target_col] *= 100

    edges.drop(columns=['overlap_sum'], inplace=True)

    return edges


def write_baseline_evals_to_db(edges, run_id, project_id, eval_type, connector, schema, table_name="baseline_evals"):
    eval_edges = edges[['source', 'target']].copy()
    eval_edges.loc[:, "id_run"] = run_id
    eval_edges.loc[:, "id_prj"] = project_id
    eval_edges.loc[:, "eval_type"] = eval_type
    eval_edges.loc[:, "eval"] = edges[eval_type]

    eval_edges.to_sql(table_name, connector, schema=schema, if_exists="append", index=False, dtype={"eval": Text()})

    return eval_edges

def write_anp_weights_to_db(limit_matrix_df, criteria_keys, metric_keys, run_id, project_id, connector, schema, table_name="anp_weights"):

    metric_weights = limit_matrix_df.iloc[len(criteria_keys):len(criteria_keys) + len(metric_keys), 0]
    metric_weights = metric_weights / metric_weights.sum()

    criteria_weights = limit_matrix_df.iloc[0:len(criteria_keys), 0]
    criteria_weights = criteria_weights / criteria_weights.sum()

    combined_weights = pd.concat([criteria_weights, metric_weights], axis=0)
    combined_weights.index = criteria_keys + metric_keys
    combined_weights = combined_weights.reset_index()
    combined_weights.columns = ['metric_or_criteria', 'weight']

    combined_weights['eval_element'] = ['criterion'] * len(criteria_keys) + ['metric'] * len(metric_keys)
    combined_weights['id_run'] = run_id
    combined_weights['id_prj'] = project_id

    combined_weights.to_sql(table_name, connector, schema=schema, if_exists="append", index=False)

    return combined_weights



def calculate_bikelane_density(edges, buffer, bike_lanes, bikelane_col):
    """Calculate bikelane density in buffered area."""

    overlaps = gpd.overlay(buffer, bike_lanes, how="intersection", keep_geom_type=False)
    overlaps['overlap_length'] = overlaps.geometry.length
    overlap_sums = overlaps.groupby('index')['overlap_length'].sum().rename('overlap_length_sum')
    edges = edges.merge(overlap_sums, on='index', how='left')
    edges[bikelane_col] = (edges['overlap_length_sum'] / (buffer['buff_area'] / 1e6)).fillna(0)
    edges = edges.drop(columns=['overlap_length_sum'])

    return edges


def calculate_land_use_mix(buffer, edges, landuse, landuse_col):
    """Calculate land use mix - Shannon's Entropy"""

    def calculate_shannon_entropy(proportions):
        return entropy(proportions, base=np.e)

    intersected = gpd.overlay(landuse, buffer, how='intersection')
    intersected['area'] = intersected.geometry.area
    land_use_areas = intersected.groupby(['index', 'typ'])['area'].sum().reset_index()
    total_area = land_use_areas.groupby('index')['area'].sum().rename('total_area')
    land_use_areas = land_use_areas.join(total_area, on='index')
    land_use_areas['proportion'] = land_use_areas['area'] / land_use_areas['total_area']

    edges_entropy = land_use_areas.groupby('index')['proportion'].apply(calculate_shannon_entropy).reset_index()
    edges_entropy = edges_entropy.rename(columns={'proportion': landuse_col})
    edges = edges.merge(edges_entropy, on='index', how='left')
    edges[landuse_col] = edges[landuse_col].fillna(0)

    return edges


def calculate_bike_lane_presence(value):
    value_map = {'P': 1, 'X': 0.75, 'L': 0.5}
    if isinstance(value, str):
        values = [value_map[char] for char in value_map if char in value]
        if values:
            return sum(values) / len(values)
        else:
            return 0

def calculate_bike_and_car_travel_time(edges, bike_speed, speed_col, bike_car_ratio_col):
    """Calculate car and bike travel speed."""
    edges['length_km'] = edges['length'] / 1000
    edges[bike_car_ratio_col] = (edges['length_km'] / int(bike_speed)) / (edges['length_km'] / edges[speed_col])
    edges = edges.drop(columns=['length_km'])

    return edges


def calculate_intersection_density(edges, buffer, intersection_col):
    """Calculate network intersection density in buffered area."""

    all_points = edges['geometry'].apply(lambda geom: [Point(geom.coords[0]), Point(geom.coords[-1])]).explode()
    point_counts = all_points.value_counts()
    degree_3_points = point_counts[point_counts == 3].index
    degree_3_points_series = gpd.GeoSeries(degree_3_points, crs=edges.crs)
    intersections = gpd.GeoDataFrame(geometry=degree_3_points_series)

    edges = merge_spatial_count(buffer, edges, intersections, intersection_col)
    edges[intersection_col] = edges[intersection_col] / (buffer['buff_area'] / 1e6)

    return edges


def enrich_network(street_graph,
                   context_datasets,
                   BIKELANE_WIDTH_COL,
                   BIKELANE_COL,
                   TRAFFIC_COL,
                   TRAFFIC_COLS,
                   TRAFFIC_PERSONAL_COL,
                   AIR_COL,
                   SLOPE_COL,
                   SPEED_COL,
                   SURFACE_COL,
                   LANDUSE_COL,
                   POP_COL, BIKE_PARKING_COL,
                   BETWEENESS_COL,
                   AVG_NODE_DEGREE_COL):

    street_graph[BIKELANE_WIDTH_COL] = pd.to_numeric(street_graph[BIKELANE_WIDTH_COL], errors='coerce').fillna(0)
    buffer = calculate_buffer(street_graph, 30)

    # BIKE LANE INFO
    street_graph['BikeLanePresence'] = street_graph[BIKELANE_COL].apply(calculate_bike_lane_presence)
    street_graph['BikeLaneWidth'] = np.where(
        (street_graph[BIKELANE_WIDTH_COL] == 0) & (street_graph['BikeLanePresence'] > 0), 1.5,
        street_graph[BIKELANE_WIDTH_COL])
    bike_lanes = street_graph[street_graph[BIKELANE_COL].str.contains("P", na=False)][['geometry', 'length']]
    street_graph = calculate_bikelane_density(street_graph, buffer, bike_lanes, 'BikeLaneDensity')

    # GREEN SPACE AND TREE CANOPY
    street_graph = merge_spatial_share(street_graph, buffer, context_datasets['green_spaces'], 'GreenSpaceShare',
                                       'buff_area', percent=True)
    street_graph['GreeneryPresence'] = np.where(street_graph['GreenSpaceShare'] > 25, 1, 0)
    street_graph = merge_spatial_share(street_graph, buffer, context_datasets['tree_canopy'], 'TreeCanopyCoverage',
                                       'buff_area', percent=True)

    # AIR QUALITY
    street_graph = merge_spatial_count(buffer, street_graph, context_datasets['air_quality'],
                                       'AirPolutantConcentration', agg_col=AIR_COL, agg_func=avg_no2)

    # TRAFFIC VOLUME
    street_graph = merge_spatial_attribute(buffer, street_graph, context_datasets['traffic_volume'], TRAFFIC_COL,
                                           'MotorisedVehicleCount')
    street_graph['MotorisedVehicleCount'] = street_graph.apply(
        lambda row: 0 if not any(char in "MHTLSRNDO" for char in row[BIKELANE_COL]) else row['MotorisedVehicleCount'],
        axis=1)
    street_graph['BusAndCarTrafficVolumeRatio'] = context_datasets['traffic_volume'][TRAFFIC_COLS].sum(axis=1) / \
                                                  context_datasets['traffic_volume'][TRAFFIC_PERSONAL_COL]
    street_graph = merge_spatial_attribute(buffer, street_graph, context_datasets['speed_limits'], SPEED_COL,
                                           'SpeedLimit')
    street_graph['MotorisedTrafficSpeed'] = street_graph['SpeedLimit'] * 0.9
    street_graph['CarLaneCount'] = street_graph[BIKELANE_COL].apply(lambda x: count_characters(str(x), "HMT"))
    street_graph = calculate_bike_and_car_travel_time(street_graph, 15, 'MotorisedTrafficSpeed',
                                                      'BikeAndCarTravelTimeRatio')

    # SLOPE AND SURFACE
    street_graph = merge_spatial_attribute(buffer, street_graph, context_datasets['slope'], SLOPE_COL, 'Slope')

    street_graph = merge_spatial_attribute(buffer, street_graph, context_datasets['surface'], SURFACE_COL,
                                           'BikeLaneSurfaceCondition')

    # LANDUSE MIX
    street_graph = calculate_land_use_mix(buffer, street_graph, context_datasets['landuse'], 'LandUseMix')
    residential = context_datasets['landuse'][context_datasets['landuse'][LANDUSE_COL] == 'residential']
    street_graph = merge_spatial_boolean(buffer, street_graph, residential, 'ResidentialAreaPresence', 'length',
                                         threshold=75)
    street_graph['ResidentialAreaPresence'] = street_graph['ResidentialAreaPresence'].astype(int)

    # POPULATION
    street_graph['PopulationDensity'] = calculate_count(buffer, context_datasets['population'], 'index', POP_COL) / (
                buffer['buff_area'] / 1e6)

    # DESTINATIONS
    street_graph['DestinationDensity'] = calculate_count(buffer, context_datasets['pois'], 'index') / (
                buffer['buff_area'] / 1e6)

    # TRANSIT
    street_graph = merge_distance_to_nearest(street_graph, context_datasets['pt_stops'], 'DistanceToTransitFacility',
                                             merge_col='index', how='left')
    street_graph['TransitFacilityDensity'] = calculate_count(buffer, context_datasets['pt_stops'], 'index') / (
                buffer['buff_area'] / 1e6)

    # BIKE PARKING
    street_graph = merge_spatial_count(buffer, street_graph, context_datasets['bike_parking'], 'BikeParkingDensity',
                                       agg_col=BIKE_PARKING_COL, agg_func="sum")

    # SINUOSITY
    street_graph['euclidean_dist'] = street_graph.geometry.apply(
        lambda geo: LineString([geo.coords[0], geo.coords[-1]]).length)
    street_graph['Sinuosity'] = street_graph['length'] / street_graph['euclidean_dist']
    street_graph['Linearity'] = street_graph['Sinuosity']

    # NETWORK
    street_graph = calculate_intersection_density(street_graph, buffer, 'IntersectionDensity')
    street_graph = merge_spatial_attribute(buffer, street_graph, context_datasets['network_centralities'],
                                           BETWEENESS_COL, 'BetweenessCentrality')
    street_graph = merge_spatial_attribute(buffer, street_graph, context_datasets['network_centralities'],
                                           AVG_NODE_DEGREE_COL, 'NodeDegree')

    return street_graph


def power_method(matrix, num_iterations=1000, tolerance=1e-10):
    n = matrix.shape[0]
    x = np.ones(n)
    for _ in range(num_iterations):
        x_new = np.dot(matrix, x)
        x_new_norm = np.linalg.norm(x_new)
        x = x_new / x_new_norm
        if np.linalg.norm(x_new - x_new_norm * x) < tolerance:
            break
    eigenvalue = np.dot(x.T, np.dot(matrix, x))
    return eigenvalue, x


def calculate_consistency_ratio(matrix):
    def get_saaty_ri(n):
        SAATY_RI = {
            1: 0.00, 2: 0.00, 3: 0.58, 4: 0.90, 5: 1.12,
            6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49
        }
        return SAATY_RI.get(n, 1.98 * (n - 2) / n)

    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Matrix must be square for consistency ratio calculation.")

    # Check for zero or negative values
    if np.any(matrix <= 0):
        raise ValueError("Matrix contains zero or negative values, which are not allowed.")

    # Use the power method to compute the dominant eigenvalue
    lambda_max, _ = power_method(matrix)
    n = matrix.shape[0]

    CI = (lambda_max - n) / (n - 1)

    if abs(CI) < 1e-10:
        CI = 0

    RI = get_saaty_ri(n)
    CR = CI / RI if RI != 0 else 0

    if abs(CR) < 1e-10:
        CR = 0

    return CR, lambda_max, CI, RI


def calculate_criteria_metric_interaction_matrix(
        metrics: pd.DataFrame,
        group_col: str,
        target_col: str
) -> pd.DataFrame:

    metric_frequency = metrics.groupby([group_col, target_col]).size().unstack(fill_value=0)
    max_freq = metric_frequency.max().max()
    min_freq = metric_frequency.min().min()
    metric_frequency = 1 + 8 * (metric_frequency - min_freq) / (max_freq - min_freq)

    criteria_to_metric = {}
    for criterion in metric_frequency.columns:
        frequencies = metric_frequency[criterion]
        n = len(frequencies)

        if n < 2:
            priority_vector = np.ones(n) / n
        else:
            matrix = np.ones((n, n))

            for i in range(n):
                for j in range(n):
                    if i != j and frequencies.iloc[i] > 0 and frequencies.iloc[j] > 0:
                        matrix[i, j] = frequencies.iloc[i] / frequencies.iloc[j]

            CR, lambda_max, CI, RI = calculate_consistency_ratio(matrix)
            if CR > 0.1:
                raise ValueError(f"Inconsistent PCM (CR = {CR:.2f}). Adjust frequency scaling.")

            priority_vector = calculate_priority_vector(matrix)

        criteria_to_metric[criterion] = priority_vector

    return pd.DataFrame(criteria_to_metric, index=metric_frequency.index)


def calculate_pairwise_comparison(
    metrics: pd.DataFrame,
    group_by_column: str,
    target_column: str
) -> tuple[np.ndarray, list]:

    freq_dict = metrics[group_by_column].value_counts().to_dict()
    min_freq = min(freq_dict.values())
    max_freq = max(freq_dict.values())
    freq_dict = {
        k: 1 + 8 * (v - min_freq) / (max_freq - min_freq) if max_freq != min_freq else 1
        for k, v in freq_dict.items()
    }

    log_freqs = {k: np.log1p(v) for k, v in freq_dict.items()}
    grouped_data = metrics.groupby(group_by_column)[target_column].apply(set).to_dict()
    sorted_keys = sorted(grouped_data.keys())
    n = len(sorted_keys)
    pcm_matrix = np.ones((n, n))

    for i, key1 in enumerate(sorted_keys):
        for j, key2 in enumerate(sorted_keys):
            if i != j:
                freq_ratio = log_freqs.get(key1, 1) / log_freqs.get(key2, 1)
                pcm_matrix[i, j] = freq_ratio
                pcm_matrix[j, i] = 1 / freq_ratio

    CR, lambda_max, CI, RI = calculate_consistency_ratio(pcm_matrix)

    if CR > 0.1:
        raise ValueError(f"Inconsistent PCM (CR = {CR:.2f}). Adjust frequency scaling.")

    return pcm_matrix, sorted_keys


def calculate_priority_vector(
        matrix: np.ndarray
) -> np.ndarray:
    # Use the power method to compute the dominant eigenvector
    _, eigenvector = power_method(matrix)
    return eigenvector / eigenvector.sum()

def get_edge_ranking(edges, metrics):
    criteria_matrix, criteria_keys = calculate_pairwise_comparison(metrics, 'criteria_type', 'metric_type')
    metric_matrix, metric_keys = calculate_pairwise_comparison(metrics, 'metric_type', 'criteria_type')
    criteria_to_metric = calculate_criteria_metric_interaction_matrix(metrics, 'metric_type', 'criteria_type')
    metric_to_criteria = calculate_criteria_metric_interaction_matrix(metrics, 'criteria_type', 'metric_type')

    edge_rankings, limit_matrix_df = perform_anp_bikeability_evaluation(edges,
                                                                        criteria_matrix,
                                                                        criteria_keys,
                                                                        metric_matrix,
                                                                        metric_keys,
                                                                        criteria_to_metric,
                                                                        metric_to_criteria)
    return edge_rankings, limit_matrix_df


def perform_anp_bikeability_evaluation(
        edges: pd.DataFrame,
        criteria_matrix: np.array,
        criteria_keys: list,
        metric_matrix: np.array,
        metric_keys: list,
        criteria_to_metric: np.array,
        metric_to_criteria: np.array
) -> tuple[np.array, pd.DataFrame]:

    edges_mcda = edges[metric_keys].dropna(how='any')
    edges_mcda_norm = normalize_minmax(edges_mcda, metric_keys)

    invert_columns = [
        "AirPolutantConcentration", "MotorisedVehicleCount", "SpeedLimit", 'CarLaneCount',
        "MotorisedTrafficSpeed", "Slope", 'DistanceToTransitFacility', 'BetweenessCentrality',
        'NodeDegree'
    ]
    invert_columns_present = [col for col in invert_columns if col in edges_mcda_norm.columns]
    edges_mcda_norm[invert_columns_present] = 1 - edges_mcda_norm[invert_columns_present]

    n = len(criteria_keys)
    m = len(metric_keys)
    r = len(edges_mcda_norm)
    supermatrix = np.zeros((n + m + r, n + m + r))

    edge_row_names = [f"{i}" for i in edges_mcda_norm.index]
    row_col_names = list(criteria_keys) + list(metric_keys) + edge_row_names

    supermatrix[:n, :n] = criteria_matrix
    supermatrix[n:n + m, :n] = criteria_to_metric
    supermatrix[:n, n:n + m] = metric_to_criteria
    supermatrix[n:n + m, n:n + m] = metric_matrix
    supermatrix[n + m:n + m + r, n:n + m] = edges_mcda_norm.values
    supermatrix[n + m:n + m + r, n + m:n + m + r] = np.identity(r)

    supermatrix_df = pd.DataFrame(supermatrix, index=row_col_names, columns=row_col_names)
    norm_supermatrix_df = supermatrix_df.div(supermatrix_df.sum(axis=0, skipna=True), axis=1)
    norm_supermatrix_df.fillna(0, inplace=True)
    limit_matrix_df = calculate_limit_matrix(norm_supermatrix_df, row_col_names)

    edge_rankings = limit_matrix_df.loc[edge_row_names, :].iloc[:, :n + m].mean(axis=1)
    edge_rankings /= edge_rankings.sum()
    edge_rankings.index = edges_mcda_norm.index
    bikeability_index = edge_rankings.fillna(0)

    return bikeability_index, limit_matrix_df


def normalize_minmax(df, columns):
    def minmax_safe(x):
        range_ = x.max() - x.min()
        if range_ == 0:
            return pd.Series(1, index=x.index)  # constant column
        return (x - x.min()) / range_

    return df[columns].apply(minmax_safe)


def calculate_limit_matrix(
        matrix: pd.DataFrame,
        row_col_names: list,
        max_iter: int = 500,
        tol: float = 1e-6
) -> Union[NDArray, DataFrame]:

    # Convert the input matrix to a sparse matrix
    sparse_matrix = csr_matrix(matrix)
    prev_matrix = sparse_matrix.copy()

    for i in range(max_iter):
        next_matrix = prev_matrix @ sparse_matrix  # Sparse matrix multiplication
        if sparse_norm(next_matrix - prev_matrix, ord='fro') < tol:
            print(f"Converged in {i + 1} iterations.")
            return pd.DataFrame(next_matrix.toarray(), index=row_col_names, columns=row_col_names)
        prev_matrix = next_matrix

    print("WARNING: Limit matrix did not fully converge.")
    return pd.DataFrame(next_matrix.toarray(), index=row_col_names, columns=row_col_names)


def filter_metrics(
        metrics: pd.DataFrame,
        occurrence: int,
        remove_columns: list
) -> pd.DataFrame:

    filtered_metrics = metrics[~metrics['criteria_type'].isna()]
    metric_n = filtered_metrics['metric_type'].value_counts()
    filtered_metrics = filtered_metrics[filtered_metrics['metric_type'].isin(metric_n[metric_n >= occurrence].index)]
    filtered_metrics = filtered_metrics[~filtered_metrics['metric_type'].str.contains('Perceived')]
    filtered_metrics = filtered_metrics[~filtered_metrics['metric_type'].isin(remove_columns)]

    return filtered_metrics