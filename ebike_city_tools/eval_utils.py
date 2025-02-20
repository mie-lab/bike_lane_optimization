import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import geopandas as gpd
from typing import Union, Any


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

    print('so far so good')
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
    df['heavy_veh'] = df[traffic_cols[1:]].sum(axis=1)

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
    buffer_30 = calculate_buffer(df[['geometry', 'index']], 30)
    buffer_15 = calculate_buffer(df[['geometry', 'index']], 15)

    bike_lanes = df[df[lane_col].str.contains("P", na=False)][['geometry', 'length']]
    df['bike_lane_density'] = calculate_count(buffer_30, bike_lanes, 'index', 'length')
    df['parks_n'] = calculate_count(buffer_30, green_spaces, 'index')
    df['housing_units'] = calculate_count(buffer_30, housing, 'index') / (buffer_30['buff_area'] / 1e6)
    df['pop_density'] = calculate_count(buffer_30, population, 'index', population_col) / (buffer_30['buff_area'] / 1e6)
    df['tree_coverage'] = (calculate_count(buffer_15, trees, 'index', 'area') / (buffer_15['buff_area'] / 1e6)) * 100

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
    df['light_ratio'] = df['street_light'] / df['length']
    df = merge_spatial_share(df, green_spaces, 'green_space_share', 'length')
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
    light_pts = assign_points(df['light_ratio'] , bins['light_ratio'])
    slope_pts = assign_points(df[slope_col] ** 2 / df['length'], bins[slope_col])
    green_space_pts = assign_points(df['green_space_share'], bins['green_space'])
    traffic_vol_pts = assign_points(df[traffic_col], bins[traffic_col])
    noise_poll_pts = assign_points(df[noise_col], bins[noise_col])
    air_quality_pts = assign_points(df['air_poll'], bins['air_poll'])

    safety = 0.27 * bike_lane_pts + 0.11 * speed_pts + 0.23 * traffic_vol_pts + 0.13 * light_pts
    comfort = 0.27 * bike_lane_pts + 0.19 * slope_pts + 0.26 * surface[surface_col].fillna(0)
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

    buff_df = df.copy()[['geometry', 'length', 'index']]
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
        other_df: gpd.GeoDataFrame,
        grouping_column: str,
        count_column: str = None
) -> gpd.GeoDataFrame:

    overlaps = gpd.sjoin(df, other_df, how='inner', predicate='intersects')

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

    if 'index' in list(spatial_data.columns):
        spatial_data = spatial_data.drop(columns=['index'])

    # Perform spatial overlay
    overlaps = gpd.overlay(buff_edges, spatial_data, how="intersection", keep_geom_type=False)
    overlaps['overlap_length'] = overlaps.geometry.length
    idx = overlaps.groupby(merge_col)['overlap_length'].idxmax()
    max_overlaps = overlaps.loc[idx]
    print(max_overlaps.columns)

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

    if overlaps.empty:
        edges[target_col] = 0
        return edges

    if edges.geometry.type[0] == 'LineString':
        overlaps['overlap'] = overlaps.geometry.length
    elif edges.geometry.type[0] == 'Polygon':
        overlaps['overlap'] = overlaps.geometry.area

    overlap_sums = overlaps.groupby(merge_col)['overlap'].sum().rename('overlap_sum')
    edges = edges.merge(overlap_sums, on=merge_col, how='left')

    edges[target_col] = (edges['overlap_sum'] / buffer[divider_col]).fillna(0)
    if percent:
        edges[target_col] *= 100

    edges.drop(columns=['overlap_sum'], inplace=True)

    return edges

