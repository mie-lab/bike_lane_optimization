import os
import geopandas as gpd
import sqlalchemy
import networkx as nx
import pandas as pd
from flask import Flask, jsonify, request
from flask_cors import CORS, cross_origin  # needs to be installed via pip install flask-cors

from sqlalchemy.orm import sessionmaker


from ebike_city_tools.graph_utils import (
    street_to_lane_graph,
    clean_street_graph_multiedges,
    clean_street_graph_directions,
    keep_only_the_largest_connected_component,
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
)
from ebike_city_tools.metrics import compute_travel_times_in_graph

# Set to True if you want to use the Database - otherwise, everything will just be saved in a dictionary
DATABASE = True
DB_LOGIN_PATH = "dblogin_ikgpgis.json"
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
maxspeed_fill_val = 50
include_lanetypes = ["H>", "H<", "M>", "M<", "M-"]
fixed_lanetypes = ["H>", "<H"]
algorithm_dict = {
    "betweenness_topdown": (topdown_betweenness_pareto, {}),
    "betweenness_cartime": (betweenness_pareto, {"betweenness_attr": "car_time"}),
    "betweenness_biketime": (betweenness_pareto, {"betweenness_attr": "bike_time"}),
}

app = Flask(__name__)
CORS(app, origins=["*", "null"])  # allowing any origin as well as localhost (null)

# load main nodes and edges that will be used for any graph
if DATABASE:
    DATABASE_CONNECTOR = get_database_connector(DB_LOGIN_PATH)
    zurich_edges = gpd.read_postgis("SELECT * FROM zurich.edges", DATABASE_CONNECTOR, geom_col="geometry").set_index(
        ["u", "v"]
    )
    zurich_nodes = gpd.read_postgis(
        "SELECT * FROM zurich.nodes", DATABASE_CONNECTOR, geom_col="geometry", index_col="osmid"
    )
    print("Loaded nodes and edges for Zurich from server", len(zurich_nodes), len(zurich_edges))
    trips_microcensus = gpd.read_postgis(
        "SELECT * FROM zurich.trips_microcensus", DATABASE_CONNECTOR, geom_col="geometry"
    )
    od_zurich = pd.read_sql("SELECT * FROM zurich.od_matrix", DATABASE_CONNECTOR)
    print("Loaded OD matrix for Zurich", len(od_zurich))
else:
    zurich_nodes = gpd.read_file(os.path.join(PATH_DATA, "street_graph_nodes.gpkg")).to_crs(CRS).set_index("osmid")
    zurich_edges = gpd.read_file(os.path.join(PATH_DATA, "street_graph_edges.gpkg")).to_crs(CRS)
    # some preprocessing
    zurich_edges = clean_street_graph_multiedges(zurich_edges)
    zurich_edges = clean_street_graph_directions(zurich_edges)

    # Load OD matrix
    # load the whole-city trip and construct origin and destination geometry
    trips_microcensus = gpd.read_file(os.path.join(PATH_DATA, "raw_od_matrix", "trips_mc_cleaned_proj.gpkg"))
    trips_microcensus["geom_destination"] = gpd.points_from_xy(
        x=trips_microcensus["end_lng"], y=trips_microcensus["end_lat"]
    )
    trips_microcensus["geom_origin"] = gpd.points_from_xy(
        x=trips_microcensus["start_lng"], y=trips_microcensus["start_lat"]
    )

    # load prebuilt OD matrix
    od_zurich = pd.read_csv(os.path.join(PATH_DATA, "od_matrix.csv"))

    # Dictionary storing the graphs and OD matrices per project. TODO: replace with database
    project_dict = {}


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
    #project_id = request.args.get("project_id", None)
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

    # Create a new project
    session = None
    try:
        Session = sessionmaker(bind=DATABASE_CONNECTOR)
        session = Session()
        cursor = session.connection().connection.cursor()
        cursor.execute(
            f"INSERT INTO webapp.projects (prj_name) VALUES ('{project_name}') RETURNING id"
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
    if DATABASE:
        
        area_polygon['id_prj'] = project_id
        area_polygon.to_postgis(
            f"bounds", DATABASE_CONNECTOR, schema=SCHEMA, if_exists="append", index=False
        )
        
        zurich_nodes_area['id_prj'] = project_id
        save_nodes = zurich_nodes_area.reset_index().rename({"osmid": "id_node"}, axis=1)[
            ["id_prj","id_node"]
        ]  # only save geometry and id
        
        save_nodes.to_sql(
            f"nodes", DATABASE_CONNECTOR, schema=SCHEMA, if_exists="append", index=False
        )
        
        # save edges for constructing the graph later
        save_edges = nx.to_pandas_edgelist(lane_graph, edge_key="edge_key")[
            ["source", "target", "edge_key", "fixed", "lanetype", "distance", "gradient", "speed_limit"]
        ]
        save_edges['id_prj'] = project_id
        save_edges.to_sql(f"edges", DATABASE_CONNECTOR, schema=SCHEMA, if_exists="append", index=False)
        
        
        # save OD matrix
        od_matrix_area_extended['id_prj'] = project_id
        od_matrix_area_extended.rename(columns={'s': 'source', 't': 'target'}, inplace=True)
        od_matrix_area_extended.to_sql(
            f"od", DATABASE_CONNECTOR, schema=SCHEMA, if_exists="append", index=False
        )
    else:
        # for debugging without using a database: save simpy in a dictionary
        if project_id is None:
            key_list = project_dict.keys()
            project_id = 0 if len(key_list) == 0 else max(key_list) + 1
        project_dict[project_id] = {"lane_graph": lane_graph, "od": od, "bounds": area_polygon}

    return (jsonify({"project_id": project_id, "variables": nr_variables, "expected_runtime": runtime_min}), 200)


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
    #run_id = request.args.get("run_id")
    algorithm = request.args.get("algorithm", "optimize")
    ratio_bike_edges = float(request.args.get("bike_ratio", "0.4"))
    optimize_every_x = float(request.args.get("optimize_frequency", "30"))
    car_weight = float(request.args.get("car_weight", "2"))
    shared_lane_factor = float(request.args.get("bike_safety_penalty", "2"))

    if DATABASE:
        try:
            edges = pd.read_sql(f"SELECT * FROM {SCHEMA}.edges WHERE id_prj = {project_id}", DATABASE_CONNECTOR)
            od = pd.read_sql(f"SELECT * FROM {SCHEMA}.od WHERE id_prj = {project_id}", DATABASE_CONNECTOR)
            
            
            edges = edges[["source", "target", "edge_key", "fixed","lanetype","distance","gradient","speed_limit"]]
            od.rename(columns={'source': 's', 'target': 't'}, inplace=True)
            od = od[["s", "t", "trips"]]
            
            # Fetch run_id from the database after insertion
            run_id = pd.read_sql(f"SELECT MAX(id_run) FROM {SCHEMA}.runs WHERE id_prj = {project_id}", DATABASE_CONNECTOR)
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
        run_list = pd.DataFrame({
            "id_prj": [project_id],
            "algorithm": [algorithm],
            "bike_ratio": [ratio_bike_edges],
            "optimize_frequency": [optimize_every_x],
            "car_weight": [car_weight],
            "bike_safety_penalty": [shared_lane_factor]
        })
    else:
        if project_id not in project_dict.keys():
            return (jsonify("Project not found. Call `construct_graph` first to start a new project"), 400)

        lane_graph = project_dict[project_id]["lane_graph"]
        od = project_dict[project_id]["od"]

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
    print("Result graph: ", result_graph)
    # convert to pandas datafrme
    result_graph_edges = nx.to_pandas_edgelist(result_graph, edge_key="edge_key")[
        ["source", "target", "edge_key", "lanetype"]
    ]
    
    result_graph_edges['id_run'] = run_id
    result_graph_edges['id_prj'] = project_id
    pareto_df['id_run'] = run_id
    pareto_df['id_prj'] = project_id   
    if DATABASE:
        run_list.to_sql(
            f"runs", DATABASE_CONNECTOR, schema=SCHEMA, if_exists="append", index=False
        )
        
        result_graph_edges.to_sql(
            f"runs_optimized", DATABASE_CONNECTOR, schema=SCHEMA, if_exists="append", index=False
        )
        pareto_df.to_sql(
            f"pareto", DATABASE_CONNECTOR, schema=SCHEMA, if_exists="append", index=False
        )
        session = None
        try:
            
            
            Session = sessionmaker(bind=DATABASE_CONNECTOR)
            session = Session()
            cursor = session.connection().connection.cursor()
            cursor.execute(
                f"""
                DROP VIEW IF EXISTS webapp.v_optimized;
                CREATE OR REPLACE VIEW webapp.v_optimized
                AS
                SELECT
                    ROW_NUMBER() OVER () AS edge_id,
                    run_opt.id_prj,
                    run_opt.id_run,
                    run_opt.source,
                    run_opt.target,
                    run_opt.edge_key,
                    run_opt.lanetype,
                    st_makeline(n1.geometry, n2.geometry) AS geometry
                FROM
                    webapp.runs_optimized run_opt
                JOIN
                    zurich.nodes n1 ON run_opt.source = n1.osmid
                JOIN
                    zurich.nodes n2 ON run_opt.target = n2.osmid
                WHERE run_opt.id_prj = {project_id} AND run_opt.id_run = {run_id};"""
            )
            session.commit()
        except Exception as e:
            if session: session.rollback()
            return (
                jsonify({"error": f"Failed to create view: {str(e)}"}),
                500,
            )
        finally:
            if session:
                session.close()
            
    else:
        project_dict[project_id][f"run{run_id}"] = result_graph_edges

    return (
        jsonify(
            {
                "project_id": project_id,
                "run_name": run_id,
                "bike_edges": len(result_graph_edges[result_graph_edges["lanetype"] == "P"]),
            }
        ),
        200,
    )


@app.route("/eval_travel_time", methods=["GET"])
def evaluate_travel_time():
    """Load the output of one run and compute the travel times
    Only works with DATABASE = True
    """
    assert DATABASE
    project_id = request.args.get("project_id")
    run_id = request.args.get("run_name")

    project_edges = pd.read_sql(f"SELECT * FROM {SCHEMA}.{project_id}_edges", DATABASE_CONNECTOR)
    project_od = pd.read_sql(f"SELECT * FROM {SCHEMA}.{project_id}_od", DATABASE_CONNECTOR)
    run_output = pd.read_sql(f"SELECT * FROM {SCHEMA}.{project_id}_run{run_id}", DATABASE_CONNECTOR)

    lane_graph = recreate_lane_graph(project_edges, run_output)

    # measure travel times
    bike_travel_time, car_travel_time = compute_travel_times_in_graph(lane_graph, project_od, SP_METHOD, WEIGHT_OD_FLOW)
    return (jsonify({"bike_travel_time": bike_travel_time, "car_travel_time": car_travel_time}), 200)


@app.route("/get_projects", methods=["GET"])
def get_projects():
    try:
        if DATABASE:
            projects = pd.read_sql("SELECT id, prj_name FROM webapp.projects", DATABASE_CONNECTOR)
            projects_json = projects.to_dict(orient="records")
            return jsonify({"projects": projects_json}), 200
        else:
            return jsonify({"error": "Database not configured"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
@app.route("/get_runs", methods=["GET"])
def get_runs():
    project_id = request.args.get("project_id")
    try:
        if DATABASE:
            runs = pd.read_sql(f"SELECT * FROM webapp.runs WHERE id_prj = {project_id}", DATABASE_CONNECTOR)
            runs_json = runs.to_dict(orient="records")
            return jsonify({"runs": runs_json}), 200
        else:
            return jsonify({"error": "Database not configured"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/create_view", methods=["POST"])
def create_view():
    # get input arguments
    project_id = request.args.get("project_id")
    run_id = request.args.get("run_id",1)
    layer = request.args.get("layer", "v_optimized")
    
    # create a view in the database
    if DATABASE:
        session = None
        try:
            Session = sessionmaker(bind=DATABASE_CONNECTOR)
            session = Session()
            cursor = session.connection().connection.cursor()
            if layer == "v_optimized":
                cursor.execute(
                    f"""
                    DROP VIEW IF EXISTS webapp.v_optimized;
                    CREATE OR REPLACE VIEW webapp.v_optimized
                    AS
                    SELECT
                        ROW_NUMBER() OVER () AS edge_id,
                        run_opt.id_prj,
                        run_opt.id_run,
                        run_opt.source,
                        run_opt.target,
                        run_opt.edge_key,
                        run_opt.lanetype,
                        st_makeline(n1.geometry, n2.geometry) AS geometry
                    FROM
                        webapp.runs_optimized run_opt
                    JOIN
                        zurich.nodes n1 ON run_opt.source = n1.osmid
                    JOIN
                        zurich.nodes n2 ON run_opt.target = n2.osmid
                    WHERE run_opt.id_prj = {project_id} AND run_opt.id_run = {run_id};"""
                )
            elif layer == "v_bound":
                cursor.execute(
                    f"""
                    DROP VIEW IF EXISTS webapp.v_bound;
                    CREATE OR REPLACE VIEW webapp.v_bound
                    AS
                    SELECT id, 
                        id_prj, 
                        ST_XMin(ST_Extent(geometry)) AS bbox_east, 
                        ST_YMin(ST_Extent(geometry)) AS bbox_south, 
                        ST_XMax(ST_Extent(geometry)) AS bbox_west, 
                        ST_YMax(ST_Extent(geometry)) AS bbox_north,
                        geometry
                    FROM webapp.bounds
                    WHERE bounds.id_prj = {project_id}
                    GROUP BY id, id_prj;"""
                    
                )
            session.commit()
            cursor.execute(
                f"""
                SELECT bbox_east, bbox_south, bbox_west, bbox_north
                FROM webapp.v_bound
                WHERE id_prj = {project_id};"""
            )
            bbox_result = cursor.fetchone()
            bbox_params = {
                "bbox_east": bbox_result[0],
                "bbox_south": bbox_result[1],
                "bbox_west": bbox_result[2],
                "bbox_north": bbox_result[3]
            }
            session.commit()
        except Exception as e:
            if session: session.rollback()
            return (
                jsonify({"error": f"Failed to create view: {str(e)}"}),
                500,
            )
        finally:
            if session:
                session.close()
            if layer == "v_bound":
                return jsonify({
                    "message": f"View created successfully",
                    "bounding_box": bbox_params
                }), 200
            else:
                return jsonify({"message": f"View created successfully"}), 200
    else:
        return jsonify({"message": "Database is not enabled"}), 400

if __name__ == "__main__":
    # run
    app.run(debug=True, host="localhost", port=8989)
