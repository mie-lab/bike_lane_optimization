import os
import geopandas as gpd
import networkx as nx
import pandas as pd
import numpy as np
import traceback
from flask import Flask, jsonify, request
from flask_cors import CORS, cross_origin  # needs to be installed via pip install flask-cors
import logging
from sqlalchemy.orm import sessionmaker

from ebike_city_tools.graph_utils import (
    street_to_lane_graph,
    keep_only_the_largest_connected_component,
    get_enriched_street_graph
)
from shapely.geometry import Polygon
from ebike_city_tools.iterative_algorithms import topdown_betweenness_pareto, betweenness_pareto
from ebike_city_tools.od_utils import extend_od_circular
from ebike_city_tools.optimize.round_optimized import ParetoRoundOptimize
from ebike_city_tools.app_utils import (
    get_database_connector,
    generate_od_nodes,
    generate_od_geometry,
    get_expected_time,
    compute_nr_variables,
    recreate_lane_graph,
    get_degree_ratios,
    get_network_bearings, recreate_lane_graph_df 
)
from ebike_city_tools.metrics import compute_travel_times_in_graph
from ebike_city_tools.eval_utils import calculate_bci, calculate_bsl, calculate_lts, calculate_blos, \
    calculate_porter_index, calculate_weikl_index, write_baseline_evals_to_db, calculate_buffer, \
    merge_spatial_attribute, calculate_bikelane_density, merge_spatial_share, merge_spatial_count, avg_no2, \
    count_characters, calculate_land_use_mix, merge_spatial_boolean, calculate_count, calculate_bike_lane_presence, \
    merge_distance_to_nearest, calculate_bike_and_car_travel_time, calculate_intersection_density, enrich_network, \
    get_edge_ranking, filter_metrics, write_anp_weights_to_db, fill_na_with_missing

# Set to True if you want to use the Database - otherwise, everything will just be saved in a dictionary
DB_LOGIN_PATH = os.path.join(os.path.dirname(__file__), "dblogin_ikgpgis.json")
SCHEMA = "webapp"
# path to load data from IF database=False:
PATH_DATA = "../street_network_data/zurich/"

# constant definitions (not designed as request arguments)
ROUNDING_METHOD = "round_bike_optimize"
IGNORE_FIXED = True
FIX_MULTILANE = False
CRS = 2056
FLOW_CONSTANT = 1  # how much flow to send through a path
SP_METHOD = "od"
WEIGHT_OD_FLOW = False
FULL_GRAPH = "_full"  # set to "" to use the version with simplified geometries
maxspeed_fill_val = 50
include_lanetypes = ["H>", "H<", "M>", "M<", "M-"]
fixed_lanetypes = ["H>", "<H"]
algorithm_dict = {
    "betweenness_topdown": (topdown_betweenness_pareto, {}),
    "betweenness_cartime": (betweenness_pareto, {"betweenness_attr": "car_time"}),
    "betweenness_biketime": (betweenness_pareto, {"betweenness_attr": "bike_time"}),
}
# contextual data columns relevant for the baseline evaluations.
SPEED_COL = 'temporeg00'
TRAFFIC_COL = 'AADT_all_veh'
TRAFFIC_PERSONAL_COL = 'AADT_personal_veh'
TRAFFIC_COLS = ['AADT_articulated_truck_veh', 'AADT_truck_veh', 'AADT_delivery_veh']
LANDUSE_COL = 'typ'
SURFACE_COL = 'belagsart'
SLOPE_COL = 'steigung_absolut'
POP_COL = 'PERS_N'
AIR_COL = 'no2'
NOISE_COL = "lre_tag"
BIKE_PARKING_COL = "anzahl_pp"
BIKELANE_WIDTH_COL = 'ln_desc_width_cycling_m'
MOTORIZED_WIDTH_COL = 'ln_desc_width_motorized_m'
BIKELANE_COL = 'ln_desc_after'
ANP_COL = 'anp'
BETWEENESS_COL = 'BetweenessCentrality'
AVG_NODE_DEGREE_COL = 'NodeDegree'

app = Flask(__name__)
CORS(app, origins=["*", "null"])  # allowing any origin as well as localhost (null)

# load main nodes and edges that will be used for any graph
db_connector = get_database_connector(DB_LOGIN_PATH)
zurich_edges = gpd.read_postgis("SELECT * FROM zurich.edges" + FULL_GRAPH, db_connector, geom_col="geometry").set_index(
    ["u", "v"]
)

zurich_nodes = gpd.read_postgis(
    "SELECT * FROM zurich.nodes" + FULL_GRAPH, db_connector, geom_col="geometry", index_col="osmid"
)
print("Loaded nodes and edges for Zurich from server", len(zurich_nodes), len(zurich_edges))

trips_microcensus = gpd.read_postgis("SELECT * FROM zurich.trips_microcensus", db_connector, geom_col="geometry")
od_zurich = pd.read_sql("SELECT * FROM zurich.od_matrix" + FULL_GRAPH, db_connector)
print("Loaded OD matrix for Zurich", len(od_zurich))

# CONTEXT DATA
queries = {
    "air_quality": "SELECT * FROM zurich.air_quality",
    "bike_parking": "SELECT * FROM zurich.bike_parking",
    "green_spaces": "SELECT * FROM zurich.green_spaces",
    "housing_units": "SELECT * FROM zurich.housing_units",
    "landuse": "SELECT * FROM zurich.landuse",
    "noise_pollution": "SELECT * FROM zurich.noise_pollution",
    "pois": "SELECT * FROM zurich.pois",
    "population": "SELECT * FROM zurich.population",
    "pt_stops": "SELECT * FROM zurich.pt_stops",
    "slope": "SELECT * FROM zurich.slope",
    "speed_limits": "SELECT * FROM zurich.speed_limits",
    "street_lighting": "SELECT * FROM zurich.street_lighting",
    "surface": "SELECT * FROM zurich.surface",
    "traffic_volume": "SELECT * FROM zurich.traffic_volume",
    "tree_canopy": "SELECT * FROM zurich.tree_canopy",
    "network_centralities": "SELECT * FROM zurich.network_centralities"
}

metrics = pd.read_sql("SELECT * FROM zurich.metrics", db_connector)
remove_columns = [
        'BuildingCondition', 'ObstaclePresence', 'DistanceToBikeLaneWithoutExclusivess',
        'SafetySupportedIntersectionShare', 'IllegalSideParkingDensity', 'SignageRatio',
        'SidewalkWidth', 'SeparatedBikeLaneDensity', 'BikeLaneType', 'RoadType',
        'BusAndCarTrafficVolumeRatio', 'BikeParkingType', 'OfficialBikeNetworkShare',
        'DetourFactor'
    ]
metrics = filter_metrics(metrics, occurrence=2, remove_columns=remove_columns)
criteria_keys = sorted(metrics['criteria_type'].unique())
metric_keys = sorted(metrics['metric_type'].unique())

# Load each dataset into a dictionary
context_datasets = {}
for name, query in queries.items():
    try:
        context_datasets[name] = gpd.read_postgis(query, db_connector, geom_col="geom")
    except Exception as e:
        print(f"Error loading {name}: {e}")

# Create and print an overview summary
overview = "\n".join([f"{name}: {len(df)} features" for name, df in context_datasets.items()])
print("Loaded contextual data for Zurich:\n" + overview)


@app.route("/get_anp", methods=["GET"])
def get_anp_evaluation():
    try:
        connector = get_database_connector(DB_LOGIN_PATH)
        project_id = int(request.args.get("project_id"))
        run_id = request.args.get("run_name")

        project_edges = pd.read_sql(f"SELECT * FROM {SCHEMA}.edges WHERE id_prj = {project_id}", connector)
        run_output = pd.read_sql(
            f"SELECT * FROM {SCHEMA}.runs_optimized WHERE id_prj = {project_id} AND id_run = {run_id}", connector)

        lane_graph = recreate_lane_graph_df(project_edges, run_output)
        street_graph = get_enriched_street_graph(lane_graph, zurich_edges)
        street_graph = enrich_network(street_graph,
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
                                      POP_COL,
                                      BIKE_PARKING_COL,
                                      BETWEENESS_COL,
                                      AVG_NODE_DEGREE_COL)

        edge_rankings, limit_matrix_df = get_edge_ranking(street_graph, metrics)
        street_graph[ANP_COL] = edge_rankings

        anp_db = write_baseline_evals_to_db(street_graph, run_id, project_id, 'anp', connector, SCHEMA)
        weights_db = write_anp_weights_to_db(limit_matrix_df, criteria_keys, metric_keys, run_id, project_id, connector,
                                         SCHEMA, table_name="anp_weights")

        return jsonify({"edges_anp": anp_db.to_dict(orient='records'), 'weights': weights_db.to_dict(orient='records')}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/get_anp_weights", methods=["GET"])
def get_anp_weights():
    try:
        connector = get_database_connector(DB_LOGIN_PATH)
        project_id = int(request.args.get("project_id"))
        run_id = int(request.args.get("run_id"))
        show_all_criteria = request.args.get("show_all_criteria", "false").lower() == "true"
        show_all_metrics = request.args.get("show_all_metrics", "false").lower() == "true"

        limit_clause_criteria = "" if show_all_criteria else "LIMIT 5"
        limit_clause_metrics = "" if show_all_metrics else "LIMIT 5"

        query_criteria = f"""
            SELECT metric_or_criteria, weight
            FROM webapp.anp_weights
            WHERE id_prj = {project_id}
              AND id_run = {run_id}
              AND eval_element = 'criterion'
            ORDER BY weight DESC
            {limit_clause_criteria}
        """

        query_metrics = f"""
            SELECT metric_or_criteria, weight
            FROM webapp.anp_weights
            WHERE id_prj = {project_id}
              AND id_run = {run_id}
              AND eval_element = 'metric'
            ORDER BY weight DESC
            {limit_clause_metrics}
        """

        df_criteria = pd.read_sql(query_criteria, connector)
        df_metrics = pd.read_sql(query_metrics, connector)

        return jsonify({
            "criteria": df_criteria.to_dict(orient="records"),
            "metrics": df_metrics.to_dict(orient="records")
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500



@app.route("/get_bci", methods=["GET"])
def get_bci_evaluation():
    try:
        connector = get_database_connector(DB_LOGIN_PATH)
        project_id = int(request.args.get("project_id"))
        run_id = request.args.get("run_name")

        project_edges = pd.read_sql(f"SELECT * FROM {SCHEMA}.edges WHERE id_prj = {project_id}", connector)
        run_output = pd.read_sql(
            f"SELECT * FROM {SCHEMA}.runs_optimized WHERE id_prj = {project_id} AND id_run = {run_id}", connector)

        lane_graph = recreate_lane_graph_df(project_edges, run_output)
        street_graph = get_enriched_street_graph(lane_graph, zurich_edges)

        edges_bci = calculate_bci(street_graph,
                                  BIKELANE_COL,
                                  BIKELANE_WIDTH_COL,
                                  MOTORIZED_WIDTH_COL,
                                  context_datasets['landuse'],
                                  LANDUSE_COL,
                                  context_datasets['traffic_volume'],
                                  TRAFFIC_COL,
                                  context_datasets['speed_limits'],
                                  SPEED_COL)
        bci_db = write_baseline_evals_to_db(edges_bci, run_id, project_id, 'bci', connector, SCHEMA)

        return jsonify({"edges_bci": bci_db.where(pd.notnull(bci_db), None).to_dict(orient='records')}), 200
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/get_bsl", methods=["GET"])
def get_bsl_evaluation():
    try:
        connector = get_database_connector(DB_LOGIN_PATH)
        project_id = int(request.args.get("project_id"))
        run_id = request.args.get("run_name")

        project_edges = pd.read_sql(f"SELECT * FROM {SCHEMA}.edges WHERE id_prj = {project_id}", connector)
        run_output = pd.read_sql(
            f"SELECT * FROM {SCHEMA}.runs_optimized WHERE id_prj = {project_id} AND id_run = {run_id}", connector)

        lane_graph = recreate_lane_graph_df(project_edges, run_output)
        street_graph = get_enriched_street_graph(lane_graph, zurich_edges)

        edges_bsl = calculate_bsl(street_graph,
                                  BIKELANE_COL,
                                  MOTORIZED_WIDTH_COL,
                                  context_datasets['speed_limits'],
                                  SPEED_COL,
                                  context_datasets['traffic_volume'],
                                  TRAFFIC_COL
                                  )
        bsl_db = write_baseline_evals_to_db(edges_bsl, run_id, project_id, 'bsl', connector, SCHEMA)

        return jsonify({"edges_bsl": bsl_db.where(pd.notnull(bsl_db), None).to_dict(orient='records')}), 200
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/get_lts", methods=["GET"])
def get_lts_evaluation():
    try:
        connector = get_database_connector(DB_LOGIN_PATH)
        project_id = int(request.args.get("project_id"))
        run_id = request.args.get("run_name")

        project_edges = pd.read_sql(f"SELECT * FROM {SCHEMA}.edges WHERE id_prj = {project_id}", connector)
        run_output = pd.read_sql(
            f"SELECT * FROM {SCHEMA}.runs_optimized WHERE id_prj = {project_id} AND id_run = {run_id}", connector)

        lane_graph = recreate_lane_graph_df(project_edges, run_output)
        street_graph = get_enriched_street_graph(lane_graph, zurich_edges)

        edges_lts = calculate_lts(street_graph,
                                  BIKELANE_COL,
                                  context_datasets['speed_limits'],
                                  SPEED_COL,
                                  context_datasets['traffic_volume'],
                                  TRAFFIC_COL
                                  )
        lts_db = write_baseline_evals_to_db(edges_lts, run_id, project_id, 'lts', connector, SCHEMA)

        return jsonify({"edges_lts": lts_db.where(pd.notnull(lts_db), None).to_dict(orient='records')}), 200
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/get_blos", methods=["GET"])
def get_blos_evaluation():
    try:
        connector = get_database_connector(DB_LOGIN_PATH)
        project_id = int(request.args.get("project_id"))
        run_id = request.args.get("run_name")

        project_edges = pd.read_sql(f"SELECT * FROM {SCHEMA}.edges WHERE id_prj = {project_id}", connector)
        run_output = pd.read_sql(
            f"SELECT * FROM {SCHEMA}.runs_optimized WHERE id_prj = {project_id} AND id_run = {run_id}", connector)

        lane_graph = recreate_lane_graph_df(project_edges, run_output)
        street_graph = get_enriched_street_graph(lane_graph, zurich_edges)

        edges_blos = calculate_blos(street_graph,
                                    BIKELANE_COL,
                                    context_datasets['traffic_volume'],
                                    TRAFFIC_COLS,
                                    MOTORIZED_WIDTH_COL,
                                    context_datasets['speed_limits'],
                                    SPEED_COL,
                                    context_datasets['surface'],
                                    SURFACE_COL
                                    )
        blos_db = write_baseline_evals_to_db(edges_blos, run_id, project_id, 'blos_grade', connector, SCHEMA)

        return jsonify({"edges_blos": blos_db.where(pd.notnull(blos_db), None).to_dict(orient='records')}), 200
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/get_porter", methods=["GET"])
def get_porter_evaluation():
    try:
        connector = get_database_connector(DB_LOGIN_PATH)
        project_id = int(request.args.get("project_id"))
        run_id = request.args.get("run_name")

        project_edges = pd.read_sql(f"SELECT * FROM {SCHEMA}.edges WHERE id_prj = {project_id}", connector)
        run_output = pd.read_sql(
            f"SELECT * FROM {SCHEMA}.runs_optimized WHERE id_prj = {project_id} AND id_run = {run_id}", connector)

        lane_graph = recreate_lane_graph_df(project_edges, run_output)
        street_graph = get_enriched_street_graph(lane_graph, zurich_edges)

        edges_porter = calculate_porter_index(
            street_graph,
            BIKELANE_COL,
            context_datasets['housing_units'],
            context_datasets['population'],
            POP_COL,
            context_datasets['green_spaces'],
            context_datasets['tree_canopy'],
            context_datasets['pt_stops'],
            context_datasets['air_quality'],
            AIR_COL
        )
        porter_db = write_baseline_evals_to_db(edges_porter, run_id, project_id, 'porter', connector, SCHEMA)

        return jsonify({"edges_porter": porter_db.where(pd.notnull(porter_db), None).to_dict(orient='records')}), 200
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/get_weikl", methods=["GET"])
def get_weikl_evaluation():
    try:
        connector = get_database_connector(DB_LOGIN_PATH)
        project_id = int(request.args.get("project_id"))
        run_id = request.args.get("run_name")

        project_edges = pd.read_sql(f"SELECT * FROM {SCHEMA}.edges WHERE id_prj = {project_id}", connector)
        run_output = pd.read_sql(
            f"SELECT * FROM {SCHEMA}.runs_optimized WHERE id_prj = {project_id} AND id_run = {run_id}", connector)

        lane_graph = recreate_lane_graph_df(project_edges, run_output)
        street_graph = get_enriched_street_graph(lane_graph, zurich_edges)

        edges_weikl = calculate_weikl_index(
            street_graph,
            BIKELANE_COL,
            context_datasets['speed_limits'],
            SPEED_COL,
            context_datasets['street_lighting'],
            context_datasets['traffic_volume'],
            TRAFFIC_COL,
            context_datasets['slope'],
            SLOPE_COL,
            context_datasets['surface'],
            SURFACE_COL,
            context_datasets['green_spaces'],
            context_datasets['noise_pollution'],
            NOISE_COL,
            context_datasets['air_quality'],
            AIR_COL
        )
        weikl_db = write_baseline_evals_to_db(edges_weikl, run_id, project_id, 'weikl', connector, SCHEMA)

        return jsonify({"edges_weikl": weikl_db.where(pd.notnull(weikl_db), None).to_dict(orient='records')}), 200
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# Set up a logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


@app.before_request
def log_request_info():
    logger.info(f"Incoming request: {request.method} {request.url}")
    logger.info(f"Headers: {dict(request.headers)}")
    if request.method in ["POST", "PUT", "PATCH"]:
        logger.info(f"Body: {request.get_data()}")


@app.route("/construct_graph", methods=["POST"])
def generate_input_graph():
    """
    Generate an input graph from a bounding polygon.
    Currently only using pre-loaded graph data from Zurich

    Notes:
    - # the input data is a list of projected 2D coordinates that can be transformed into a Polygon, e.g.
        [[2678000.0, 1247000.0], [2678000.0, 1250000.0], [2681000.0, 1250000.0], [2681000.0, 1247000.0]]
    - In future versions, this should be extended to constructing arbitrary graphs from OSM data.
    - There are two modes for creating the OD matrix, one simply based on the nodes (named "fast") and one based on the
        geometries ("slow")
    """
    # get input arguments: polygons for bounding the region and OD mode
    bounds_polygon = request.get_json(force=True)
    od_creation_mode = request.args.get("odmode", "fast")
    # project_id = request.args.get("project_id", None)
    project_name = request.args.get("project_name", None)

    try:
        area_polygon = Polygon(bounds_polygon)
    except ValueError:
        return (jsonify("Coordinates have wrong format. Check the documentation."), 400)
    area_polygon = gpd.GeoDataFrame(geometry=[area_polygon], crs=CRS)

    # restrict graph to Polygon
    zurich_nodes_area = zurich_nodes.sjoin(area_polygon)
    zurich_edges_area = zurich_edges.sjoin(area_polygon)
    # if the graph is empty, return message
    if len(zurich_edges_area) == 0:
        return (jsonify("No edges found in this area. Try with other coordinates."), 400)

    # create OD matrix
    if od_creation_mode == "fast":
        od = generate_od_nodes(od_zurich, zurich_nodes_area)
    elif od_creation_mode == "slow":
        od = generate_od_geometry(area_polygon, trips_microcensus, zurich_nodes_area)
    else:
        return (jsonify("Wrong value for odmode argument. Must be one of {slow, fast}"), 400)

    # create graph
    lane_graph = street_to_lane_graph(
        zurich_nodes_area,
        zurich_edges_area,
        maxspeed_fill_val=maxspeed_fill_val,
        include_lanetypes=include_lanetypes,
        fixed_lanetypes=fixed_lanetypes,
        target_crs=CRS,
    )
    # reduce to largest connected component
    lane_graph = keep_only_the_largest_connected_component(lane_graph)

    # we need to extend the OD matrix to guarantee connectivity of the car network
    od = od[od["s"] != od["t"]]
    node_list = list(lane_graph.nodes())
    od = od[(od["s"].isin(node_list)) & (od["t"].isin(node_list))]
    od_matrix_area_extended = extend_od_circular(od, node_list)

    # estimate runtime
    nr_variables = compute_nr_variables(lane_graph.number_of_edges(), len(od_matrix_area_extended))
    runtime_min = get_expected_time(nr_variables)

    # save nodes for the geometry

    connector = get_database_connector(DB_LOGIN_PATH)
    # Create a new project
    session = None
    try:
        Session = sessionmaker(bind=connector)
        session = Session()
        cursor = session.connection().connection.cursor()
        cursor.execute(
            f"INSERT INTO webapp.projects (prj_name, runtime_min) VALUES ('{project_name}', {np.round(runtime_min,2)}) RETURNING id"
        )
        project_id = cursor.fetchone()[0]
        session.commit()
    except Exception as e:
        if session:
            session.rollback()
        return jsonify({"error": str(e)}), 500
    finally:
        if session:
            session.close()

    area_polygon["id_prj"] = project_id
    area_polygon.to_postgis(f"bounds", connector, schema=SCHEMA, if_exists="append", index=False)

    zurich_nodes_area["id_prj"] = project_id
    save_nodes = zurich_nodes_area.reset_index().rename({"osmid": "id_node"}, axis=1)[
        ["id_prj", "id_node"]
    ]  # only save geometry and id

    save_nodes.to_sql(f"nodes", connector, schema=SCHEMA, if_exists="append", index=False)

    # save edges for constructing the graph later
    save_edges = nx.to_pandas_edgelist(lane_graph, edge_key="edge_key")[
        ["source", "target", "edge_key", "fixed", "lanetype", "distance", "gradient", "speed_limit"]
    ]
    save_edges["id_prj"] = project_id
    save_edges.to_sql(f"edges", connector, schema=SCHEMA, if_exists="append", index=False)

    # save OD matrix
    od_matrix_area_extended["id_prj"] = project_id
    od_matrix_area_extended.rename(columns={"s": "source", "t": "target"}, inplace=True)
    od_matrix_area_extended.to_sql(f"od", connector, schema=SCHEMA, if_exists="append", index=False)

    return (jsonify({"project_id": project_id, "variables": nr_variables, "expected_runtime": runtime_min}), 200)


@app.route("/get_new_run_id", methods=["GET"])
def get_new_run_id():
    project_id = int(request.args.get("project_id"))
    connector = get_database_connector(DB_LOGIN_PATH)
    try:
        run_id = pd.read_sql(f"SELECT MAX(id_run) FROM {SCHEMA}.runs WHERE id_prj = {project_id}", connector)
        if run_id.empty or run_id.iloc[0][0] is None:
            run_id = 1
        else:
            run_id = int(run_id.iloc[0][0]) + 1
        return jsonify({"run_id": run_id}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/optimize", methods=["GET", "POST"])
def optimize():
    """
    Run optimization on graph that was previously created
    Request arguments:
        project_id: Name of the project, defined by one specific area. The data must be preloaded
        run_name: Name of the specific run, defined by a set of parameters
        algorithm: One of {optimize, betweenness_biketime, betweenness_cartime, betweenness_topdown} - see paper
        ratio_bike_edges: How many lanes should preferably become bike lanes? Defaults to 0.4 --> 40% of all lanes
        optimize_every_x: How often to re-run the optimization. The more often, the better, but also much slower
        car_weight: Weighting of the car travel time in the objective function. Should be something between 0.1 and 10
        bike_safety_penalty: factor by how much the perceived bike travel time increases if cycling on car lane.
            Defaults to 2, i.e. the perceived travel time on a car lane is twice as much as the one on a bike lane
    e.g. test with
    curl -X GET "http://localhost:8989/optimize?project_id=test&algorithm=betweenness_biketime&run_name=1&bike_ratio=0.1"
    """
    project_id = int(request.args.get("project_id"))
    run_name = request.args.get("run_name", "")
    algorithm = request.args.get("algorithm", "optimize")
    ratio_bike_edges = float(request.args.get("bike_ratio", "0.4"))
    optimize_every_x = float(request.args.get("optimize_frequency", "30"))

    car_weight = float(request.args.get("car_weight", "0.7"))
    shared_lane_factor = float(request.args.get("bike_safety_penalty", "2"))

    connector = get_database_connector(DB_LOGIN_PATH)
    try:
        edges = pd.read_sql(f"SELECT * FROM {SCHEMA}.edges WHERE id_prj = {project_id}", connector)
        od = pd.read_sql(f"SELECT * FROM {SCHEMA}.od WHERE id_prj = {project_id}", connector)

        edges = edges[["source", "target", "edge_key", "fixed", "lanetype", "distance", "gradient", "speed_limit"]]
        edges["capacity"] = 1  # since it's a lane graph with one edge per lane, every edge has capacity 1
        od.rename(columns={"source": "s", "target": "t"}, inplace=True)
        od = od[["s", "t", "trips"]]

        # Fetch run_id from the database after insertion
        run_id = pd.read_sql(f"SELECT MAX(id_run) FROM {SCHEMA}.runs WHERE id_prj = {project_id}", connector)
        print("run Id: --> ", run_id.iloc[0][0])
        if run_id.empty or run_id.iloc[0][0] is None:
            run_id = 1
        else:
            run_id = int(run_id.iloc[0][0]) + 1
    except:
        return (
            jsonify("Problem loading project from database. To start a new project, call `construct_graph` first"),
            400,
        )

    lane_graph = nx.from_pandas_edgelist(
        edges,
        edge_key="edge_key",
        edge_attr=[col for col in edges.columns if col not in ["source", "target", "edge_key"]],
        create_using=nx.MultiDiGraph,
    )
    run_list = pd.DataFrame(
        {
            "id_prj": [project_id],
            "algorithm": [algorithm],
            "bike_ratio": [ratio_bike_edges],
            "optimize_frequency": [optimize_every_x],
            "car_weight": [car_weight],
            "bike_safety_penalty": [shared_lane_factor],
            "run_name": [run_name],
        }
    )

    # compute the absolute number of bike lanes that are desired
    desired_edge_count = int(ratio_bike_edges * lane_graph.number_of_edges())
    print("Desired edges", desired_edge_count, lane_graph.number_of_edges(), len(od))

    if "betweenness" in algorithm:

        print(f"Running betweenness algorithm {algorithm}")
        # get algorithm method
        algorithm_func, kwargs = algorithm_dict[algorithm]

        # run betweenness centrality algorithm for comparison
        result_graph, pareto_df = algorithm_func(
            lane_graph.copy(),
            sp_method=SP_METHOD,
            od_matrix=od,
            weight_od_flow=WEIGHT_OD_FLOW,
            fix_multilane=FIX_MULTILANE,
            shared_lane_factor=shared_lane_factor,
            save_graph_path=None,
            return_graph_at_edges=desired_edge_count,
            **kwargs,
        )
    else:
        # potentially only extend OD matrix for the optimzation algorithm and not for the betweenness algorithm
        # od = extend_od_circular(od, node_list)

        opt = ParetoRoundOptimize(
            lane_graph.copy(),
            od.copy(),
            optimize_every_x=optimize_every_x,
            car_weight=car_weight,
            sp_method=SP_METHOD,
            shared_lane_factor=shared_lane_factor,
            weight_od_flow=WEIGHT_OD_FLOW,
            valid_edges_k=0,
        )
        # RUN pareto optimization, potentially with saving the graph after each optimization step
        result_graph, pareto_df = opt.pareto(fix_multilane=FIX_MULTILANE, return_graph_at_edges=desired_edge_count)
    # convert to pandas datafrme
    result_graph_edges = nx.to_pandas_edgelist(result_graph, edge_key="edge_key")[
        ["source", "target", "edge_key", "lanetype"]
    ]

    result_graph_edges["id_run"] = run_id
    result_graph_edges["id_prj"] = project_id
    pareto_df["id_run"] = run_id
    pareto_df["id_prj"] = project_id

    # compute relative timees
    base_bike, base_car = pareto_df["bike_time"].max(), pareto_df["car_time"].min()
    pareto_df["car_time_change"] = (pareto_df["car_time"] - base_car) / base_car * 100
    pareto_df["bike_time_change"] = (pareto_df["bike_time"] - base_bike) / base_bike * 100

    connector = get_database_connector(DB_LOGIN_PATH)
    run_list.to_sql(f"runs", connector, schema=SCHEMA, if_exists="append", index=False)

    result_graph_edges.to_sql(f"runs_optimized", connector, schema=SCHEMA, if_exists="append", index=False)
    pareto_df.to_sql(f"pareto", connector, schema=SCHEMA, if_exists="append", index=False)

    return (
        jsonify(
            {
                "project_id": project_id,
                "run_id": run_id,
                "run_name": run_name,
                "bike_edges": len(result_graph_edges[result_graph_edges["lanetype"] == "P"]),
            }
        ),
        200,
    )


@app.route("/get_distance_per_lane_type", methods=["GET"])
def get_distance_per_lane_type():
    try:
        project_id = int(request.args.get("project_id"))
        run_id = request.args.get("run_name")

        connector = get_database_connector(DB_LOGIN_PATH)
        bike_distance = pd.read_sql(
            f"SELECT SUM(edges.distance) AS total_bike_lane_distance FROM {SCHEMA}.runs_optimized JOIN {SCHEMA}.edges ON runs_optimized.source = edges.source AND runs_optimized.target = edges.target WHERE runs_optimized.lanetype = 'P' AND runs_optimized.id_run ={run_id} AND runs_optimized.id_prj = {project_id}",
            connector,
        )
        car_distance = pd.read_sql(
            f"SELECT SUM(edges.distance) AS total_car_lane_distance FROM {SCHEMA}.runs_optimized JOIN {SCHEMA}.edges ON runs_optimized.source = edges.source AND runs_optimized.target = edges.target WHERE runs_optimized.lanetype = 'M>' AND runs_optimized.id_run ={run_id} AND runs_optimized.id_prj = {project_id}",
            connector,
        )
        bike_distance_json = bike_distance.to_dict(orient="records")
        car_distance_json = car_distance.to_dict(orient="records")

        return (jsonify({"distance_bike": bike_distance_json, "distance_car": car_distance_json}), 200)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/eval_travel_time", methods=["GET"])
def evaluate_travel_time():
    """
    Load the output of one run and compute the travel times
    """
    project_id = request.args.get("project_id")
    run_id = request.args.get("run_name")

    connector = get_database_connector(DB_LOGIN_PATH)
    project_edges = pd.read_sql(f"SELECT * FROM {SCHEMA}.edges WHERE id_prj = {project_id}", connector)
    project_od = pd.read_sql(f"SELECT * FROM {SCHEMA}.od WHERE id_prj = {project_id}", connector)
    run_output = pd.read_sql(
        f"SELECT * FROM {SCHEMA}.runs_optimized WHERE id_prj = {project_id} AND id_run = {run_id}", connector
    )

    # put lanetype attribute from run_output onto the edges and update bike and car travel time attributes
    lane_graph = recreate_lane_graph(project_edges, run_output)

    # rename columns
    project_edges.rename(columns={"source": "s", "target": "t"}, inplace=True)
    project_od.rename(columns={"source": "s", "target": "t"}, inplace=True)
    run_output.rename(columns={"source": "s", "target": "t"}, inplace=True)

    # measure travel times -> TODO: would be better to just use the pareto times, and do some other evaluation here
    bike_travel_time, car_travel_time = compute_travel_times_in_graph(lane_graph, project_od, SP_METHOD, WEIGHT_OD_FLOW)
    return (jsonify({"bike_travel_time": bike_travel_time, "car_travel_time": car_travel_time}), 200)


@app.route("/get_pareto", methods=["GET"])
def get_pareto():
    try:
        connector = get_database_connector(DB_LOGIN_PATH)
        project_id = request.args.get("project_id")
        run_id = request.args.get("run_name")

        pareto = pd.read_sql(
            f"SELECT * FROM {SCHEMA}.pareto WHERE id_prj = {project_id} AND id_run = {run_id}", connector
        )
        pareto_json = pareto.to_dict(orient="records")
        return jsonify({"projects": pareto_json}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/get_complexity", methods=["GET"])
def get_complexity():
    try:
        project_id = int(request.args.get("project_id"))
        run_id = request.args.get("run_name")

        connector = get_database_connector(DB_LOGIN_PATH)
        project_edges = pd.read_sql(f"SELECT * FROM {SCHEMA}.edges WHERE id_prj = {project_id}", connector)
        run_output = pd.read_sql(
            f"SELECT * FROM {SCHEMA}.runs_optimized WHERE id_prj = {project_id} AND id_run = {run_id}", connector
        )

        lane_graph = recreate_lane_graph(project_edges, run_output)

        bike_degree_ratios = get_degree_ratios(lane_graph, "P")
        car_degree_ratios = get_degree_ratios(lane_graph, "M")

        return (jsonify({"bike_degree_ratio": bike_degree_ratios, "car_degree_ratios": car_degree_ratios}), 200)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/get_network_bearing", methods=["GET"])
def get_network_bearing():
    try:
        project_id = int(request.args.get("project_id"))
        run_id = request.args.get("run_name")

        connector = get_database_connector(DB_LOGIN_PATH)
        project_edges = pd.read_sql(f"SELECT * FROM {SCHEMA}.edges WHERE id_prj = {project_id}", connector)
        run_output = pd.read_sql(
            f"SELECT * FROM {SCHEMA}.runs_optimized WHERE id_prj = {project_id} AND id_run = {run_id}", connector
        )

        # load nodes from database
        nodes_zurich = pd.read_sql(
            f"""
                SELECT z.osmid, z.x, z.y
                FROM zurich.nodes{FULL_GRAPH} AS z
                JOIN  webapp.edges AS w ON w.source = z.osmid OR w.target = z.osmid
                WHERE w.id_prj = {project_id}
                """,
            connector,
        )

        lane_graph = recreate_lane_graph(project_edges, run_output)

        xs = {nodes_zurich.loc[i, "osmid"]: float(nodes_zurich.loc[i, "x"]) for i in nodes_zurich.index}
        nx.set_node_attributes(lane_graph, xs, "x")

        ys = {nodes_zurich.loc[i, "osmid"]: float(nodes_zurich.loc[i, "y"]) for i in nodes_zurich.index}
        nx.set_node_attributes(lane_graph, ys, "y")

        lane_graph.graph["crs"] = 4326

        bike_network_bearings = get_network_bearings(lane_graph, "P", "distance")
        car_network_bearings = get_network_bearings(lane_graph, "M", "distance")

        return (
            jsonify({"bike_network_bearings": bike_network_bearings, "car_network_bearings": car_network_bearings}),
            200,
        )

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/get_projects", methods=["GET"])
def get_projects():
    try:
        connector = get_database_connector(DB_LOGIN_PATH)
        projects = pd.read_sql("SELECT id, prj_name, created, runtime_min FROM webapp.projects", connector)
        replaced_df = projects.replace({np.nan: None})
        projects_json = replaced_df.to_dict(orient="records")
        return jsonify({"projects": projects_json}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/get_runs", methods=["GET"])
def get_runs():
    project_id = request.args.get("project_id")
    try:
        connector = get_database_connector(DB_LOGIN_PATH)
        runs = pd.read_sql(f"SELECT * FROM webapp.runs WHERE id_prj = {project_id}", connector)
        runs_json = runs.to_dict(orient="records")
        return jsonify({"runs": runs_json}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/check_eval_metric_exists", methods=["GET"])
def check_eval_metric_exists():
    try:
        connector = get_database_connector(DB_LOGIN_PATH)
        project_id = int(request.args.get("project_id"))
        run_id = int(request.args.get("run_id"))
        metric_key = request.args.get("metric_key")
        query = f"""
            SELECT 1
            FROM {SCHEMA}.baseline_evals
            WHERE id_prj = {project_id}
              AND id_run = {run_id}
              AND eval_type = '{metric_key}'
            LIMIT 1;
        """
        result = pd.read_sql(query, connector)
        exists = not result.empty
        return jsonify({"exists": exists}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/getBoundingBox", methods=["GET"])
def get_bounding_box():
    project_id = request.args.get("project_id")
    bbox_params = None  # Initialize bbox_params to None

    session = None

    sql_statement = f"""
        SELECT bbox_east, bbox_south, bbox_west, bbox_north
        FROM webapp.v_bound
        WHERE id_prj = {project_id};"""

    try:
        connector = get_database_connector(DB_LOGIN_PATH)
        Session = sessionmaker(bind=connector)
        session = Session()
        cursor = session.connection().connection.cursor()

        cursor.execute(sql_statement)

        bbox_result = cursor.fetchone()
        if bbox_result:
            bbox_params = {
                "bbox_east": bbox_result[0],
                "bbox_south": bbox_result[1],
                "bbox_west": bbox_result[2],
                "bbox_north": bbox_result[3],
            }
    except Exception as e:
        if session:
            session.rollback()
        return jsonify({"error": f"Failed to get Bounding Box: {str(e)}"}), 500
    finally:
        if session:
            session.close()

        if bbox_params:
            return jsonify({"message": "Bounding box retrieved successfully", "bounding_box": bbox_params}), 200
        else:
            return jsonify({"message": "No bounding box found"}), 200


def create_app():
    return app


if __name__ == "__main__":
    # run
    app.run(debug=True, host="0.0.0.0", port=8989)
