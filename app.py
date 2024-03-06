import os
import numpy as np
import geopandas as gpd
import pyproj
import networkx as nx

from flask import Flask, jsonify, request
from flask_cors import CORS, cross_origin  # needs to be installed via pip install flask-cors

from ebike_city_tools.graph_utils import (
    street_to_lane_graph,
    clean_street_graph_multiedges,
    clean_street_graph_directions,
    lane_to_street_graph,
)
from shapely.geometry import Polygon
from ebike_city_tools.iterative_algorithms import topdown_betweenness_pareto, betweenness_pareto
from ebike_city_tools.od_utils import extend_od_circular
from ebike_city_tools.optimize.round_optimized import ParetoRoundOptimize
from ebike_city_tools.app_utils import (
    PATH_DATA,
    generate_od_nodes,
    generate_od_geometry,
    get_expected_time,
    compute_nr_variables,
)

maxspeed_fill_val = 50
include_lanetypes = ["H>", "H<", "M>", "M<", "M-"]
fixed_lanetypes = ["H>", "<H"]

ROUNDING_METHOD = "round_bike_optimize"
IGNORE_FIXED = True
FIX_MULTILANE = False
FLOW_CONSTANT = 1  # how much flow to send through a path
OPTIMIZE_EVERY_K = 50
CAR_WEIGHT = 2
SP_METHOD = "od"
SHARED_LANE_FACTOR = 2
WEIGHT_OD_FLOW = False
RATIO_BIKE_EDGES = 0.4
algorithm_dict = {
    "betweenness_topdown": (topdown_betweenness_pareto, {}),
    "betweenness_cartime": (betweenness_pareto, {"betweenness_attr": "car_time"}),
    "betweenness_biketime": (betweenness_pareto, {"betweenness_attr": "bike_time"}),
}


app = Flask(__name__)
CORS(app, origins=["*", "null"])  # allowing any origin as well as localhost (null)

# load main nodes and edges that will be used for any graph
CRS = 2056
zurich_nodes = gpd.read_file(os.path.join(PATH_DATA, "street_graph_nodes.gpkg")).to_crs(CRS).set_index("osmid")
zurich_edges = gpd.read_file(os.path.join(PATH_DATA, "street_graph_edges.gpkg")).to_crs(CRS)
# some preprocessing
zurich_edges = clean_street_graph_multiedges(zurich_edges)
zurich_edges = clean_street_graph_directions(zurich_edges)

# Dictionary storing the graphs and OD matrices per project. TODO: replace with database
project_dict = {}


def create_new_project_id():
    key_list = project_dict.keys()
    if len(key_list) == 0:
        return 0
    else:
        return max(key_list) + 1


@app.route("/construct_graph", methods="POST")
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

    try:
        area_polygon = Polygon(bounds_polygon, crs=CRS)
    except ValueError:
        return (jsonify("Coordinates have wrong format. Check the documentation."), 400)

    # restrict graph to Polygon
    zurich_nodes_area = zurich_nodes.sjoin(area_polygon)
    zurich_edges_area = zurich_edges.sjoin(area_polygon)
    # if the graph is empty, return message
    if len(zurich_edges_area) == 0:
        return (jsonify("No edges found in this area. Try with other coordinates."), 400)

    # create OD matrix
    if od_creation_mode == "fast":
        od = generate_od_nodes(zurich_nodes_area)
    elif od_creation_mode == "slow":
        od = generate_od_geometry(area_polygon, zurich_nodes_area)
    else:
        return (jsonify("Wrong value for odmode argument. Must be one of {slow, fast}"), 400)

    # create graph
    G_lane = street_to_lane_graph(
        zurich_nodes_area,
        zurich_edges_area,
        maxspeed_fill_val=maxspeed_fill_val,
        include_lanetypes=include_lanetypes,
        fixed_lanetypes=fixed_lanetypes,
        target_crs=CRS,
    )

    # we need to extend the OD matrix to guarantee connectivity of the car network
    od = od[od["s"] != od["t"]]
    node_list = list(G_lane.nodes())
    od = od[(od["s"].isin(node_list)) & (od["t"].isin(node_list))]
    od_matrix_area_extended = extend_od_circular(od, node_list)

    # estimate runtime
    nr_variables = compute_nr_variables(G_lane.number_of_edges(), len(od_matrix_area_extended))
    runtime_min = get_expected_time(nr_variables)

    # save graph and OD matrix with a project ID
    project_id = create_new_project_id()
    project_dict[project_id] = {"lane_graph": G_lane, "od": od, "bounds": area_polygon}

    return (jsonify({"project_id": project_id, "variables": nr_variables, "expected_runtime": runtime_min}), 200)


@app.route("/optimize", methods=["POST"])
def optimize():
    """
    Run optimization on graph that was previously created
    Request arguments:
        algorithm: One of {optimize, betweenness_biketime, betweenness_cartime, betweenness_topdown} - see paper
        ratio_bike_edges: How many lanes should preferably become bike lanes? Defaults to 0.4 --> 40% of all lanes
        optimize_every_x: How often to re-run the optimization. The more often, the better, but also much slower
        car_weight: Weighting of the car travel time in the objective function. Should be something between 0.1 and 10
        bike_safety_penalty: factor by how much the perceived bike travel time increases if cycling on car lane.
            Defaults to 2, i.e. the perceived travel time on a car lane is twice as much as the one on a bike lane
    """
    project_id = request.args.get("project_id")
    algorithm = request.args.get("algorithm", "optimize")
    ratio_bike_edges = float(request.args.get("bike_ratio", "0.4"))
    optimize_every_x = float(request.args.get("optimize_frequency", "30"))
    car_weight = float(request.args.get("car_weight", "2"))
    shared_lane_factor = float(request.args.get("bike_safety_penalty", "2"))

    if project_id not in project_dict.keys():
        return (jsonify("Project not find. Call `construct_graph` first to start a new project"), 400)

    G_lane = project_dict[project_id]["lane_graph"]
    od = project_dict[project_id]["od"]

    # compute the absolute number of bike lanes that are desired
    desired_edge_count = int(ratio_bike_edges * G_lane.number_of_edges())

    if "betweenness" in algorithm:

        print(f"Running betweenness algorithm {algorithm}")
        # get algorithm method
        algorithm_func, kwargs = algorithm_dict[algorithm]

        # run betweenness centrality algorithm for comparison
        result_graph = algorithm_func(
            G_lane.copy(),
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
            G_lane.copy(),
            od.copy(),
            optimize_every_x=optimize_every_x,
            car_weight=car_weight,
            sp_method=SP_METHOD,
            shared_lane_factor=shared_lane_factor,
            weight_od_flow=WEIGHT_OD_FLOW,
            valid_edges_k=0,
        )
        # RUN pareto optimization, potentially with saving the graph after each optimization step
        result_graph = opt.pareto(fix_multilane=FIX_MULTILANE, return_graph_at_edges=desired_edge_count)

    # convert to pandas datafrme
    result_graph_edges = nx.to_pandas_edgelist(result_graph)

    # TODO: modify output to return only geometries with bike/car label
    return jsonify(result_graph_edges)


if __name__ == "__main__":
    # run
    app.run(debug=True, host="localhost", port=8989)
