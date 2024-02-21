import numpy as np

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
from ebike_city_tools.iterative_algorithms import topdown_betweenness_pareto, betweenness_pareto
from ebike_city_tools.od_utils import extend_od_circular
from ebike_city_tools.optimize.round_optimized import ParetoRoundOptimize

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


@app.route("/optimize", methods=["POST"])
def optimize():

    data = request.get_json(force=True)
    algorithm = request.args.get("algorithm", "optimize")
    ratio_bike_edges = float(request.args.get("bike_ratio", "0.4"))

    # ARGUMENTS --> TODO: get from request args instead
    remove_multistreets = True
    target_crs = 2056
    maxspeed_fill_val = 50
    include_lanetypes = ["H>", "H<", "M>", "M<", "M-"]
    fixed_lanetypes = ["H>", "<H"]

    desired_edge_count = int(ratio_bike_edges * G_lane.number_of_edges())

    # TODO: extract nodes and edges from transmitted data
    # street_graph_nodes, street_graph_edges = ...
    # od = ...
    if remove_multistreets:
        street_graph_edges = clean_street_graph_multiedges(street_graph_edges)
        print("removed multiedges from street graph", len(street_graph_edges))
        street_graph_edges = clean_street_graph_directions(street_graph_edges)
        print("removed bidirectional edges from street graph", len(street_graph_edges))

    # convert to lane graph
    G_lane = street_to_lane_graph(
        street_graph_nodes,
        street_graph_edges,
        maxspeed_fill_val=maxspeed_fill_val,
        include_lanetypes=include_lanetypes,
        fixed_lanetypes=fixed_lanetypes,
        target_crs=target_crs,
    )

    # TODO: use only Zurich and load OD from file
    od = od[od["s"] != od["t"]]
    # reduce OD matrix to nodes that are in G_lane
    node_list = list(G_lane.nodes())
    od = od[(od["s"].isin(node_list)) & (od["t"].isin(node_list))]

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
            save_graph_path=None,
            return_graph_at_edges=desired_edge_count,
            **kwargs,
        )
    else:
        # FOR OPTIMIZATION
        # extend OD matrix because otherwise we get disconnected car graph
        od = extend_od_circular(od, node_list)

        opt = ParetoRoundOptimize(
            G_lane.copy(),
            od.copy(),
            optimize_every_x=OPTIMIZE_EVERY_K,
            car_weight=CAR_WEIGHT,
            sp_method=SP_METHOD,
            shared_lane_factor=SHARED_LANE_FACTOR,
            weight_od_flow=WEIGHT_OD_FLOW,
            valid_edges_k=0,
        )
        # RUN pareto optimization, potentially with saving the graph after each optimization step
        result_graph = opt.pareto(fix_multilane=FIX_MULTILANE, return_graph_at_edges=desired_edge_count)

    # convert to pandas datafrme
    result_graph_edges = nx.to_pandas_edgelist(result_graph)
    return jsonify(result_graph_edges)


if __name__ == "__main__":
    # run
    app.run(debug=True, host="localhost", port=8989)
