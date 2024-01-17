import time
import os
import json
import argparse
import pandas as pd
import numpy as np
import geopandas as gpd
import networkx as nx

from ebike_city_tools.optimize.linear_program import define_IP
from ebike_city_tools.utils import (
    lane_to_street_graph,
    extend_od_circular,
    output_to_dataframe,
    output_lane_graph,
)
from ebike_city_tools.optimize.rounding_utils import combine_paretos_from_path, combine_pareto_frontiers
from ebike_city_tools.optimize.round_simple import pareto_frontier, rounding_and_splitting
from ebike_city_tools.iterative_algorithms import betweenness_pareto, topdown_betweenness_pareto
from ebike_city_tools.optimize.wrapper import adapt_edge_attributes
from ebike_city_tools.optimize.round_optimized import ParetoRoundOptimize

from snman import distribution, street_graph, graph, io, merge_edges, lane_graph
from snman.constants import (
    KEY_LANES_DESCRIPTION,
    KEY_LANES_DESCRIPTION_AFTER,
    MODE_PRIVATE_CARS,
    KEY_GIVEN_LANES_DESCRIPTION,
)

ROUNDING_METHOD = "round_bike_optimize"
IGNORE_FIXED = True
FLOW_CONSTANT = 1  # how much flow to send through a path
WEIGHT_OD_FLOW = False
RATIO_BIKE_EDGES = 0.4
algorithm_dict = {
    "betweenness_topdown": (topdown_betweenness_pareto, {}),
    "betweenness_cartime": (betweenness_pareto, {"betweenness_attr": "car_time"}),
    "betweenness_biketime": (betweenness_pareto, {"betweenness_attr": "bike_time"}),
}

od_quantile_used = {"chicago": 0.9, "cambridge": 0.5}  # set this with the final parameters


def generate_motorized_lane_graph(
    edge_path,
    node_path,
    source_lanes_attribute=KEY_LANES_DESCRIPTION,
    target_lanes_attribute=KEY_LANES_DESCRIPTION_AFTER,
    return_H=False,
):
    G = io.load_street_graph(edge_path, node_path)  # initialize lanes after rebuild
    print("Initial street graph edges:", G.number_of_edges())
    # need to save the maxspeed attribute here to use it later
    nx.set_edge_attributes(G, nx.get_edge_attributes(G, source_lanes_attribute), target_lanes_attribute)
    # ensure consistent edge directions (only from lower to higher node!)
    street_graph.organize_edge_directions(G)

    distribution.set_given_lanes(G)
    H = street_graph.filter_lanes_by_modes(G, {MODE_PRIVATE_CARS}, lane_description_key=KEY_GIVEN_LANES_DESCRIPTION)

    merge_edges.reset_intermediate_nodes(H)
    merge_edges.merge_consecutive_edges(H, distinction_attributes={KEY_LANES_DESCRIPTION_AFTER})
    # make lane graph
    L = lane_graph.create_lane_graph(H, KEY_GIVEN_LANES_DESCRIPTION)
    print("Initial lane graph edges:", L.number_of_edges())
    # make sure that the graph is strongly connected
    L = graph.keep_only_the_largest_connected_component(L)
    print("Lane graph edges after keeping connected component:", L.number_of_edges())
    # add some edge attributes that we need for the optimization (e.g. slope)
    L = adapt_edge_attributes(L, ignore_fixed=IGNORE_FIXED)
    if return_H:
        return H, L
    return L


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data_path", default="../street_network_data/zollikerberg", type=str)
    parser.add_argument("-i", "--instance", default="affoltern", type=str)
    parser.add_argument("-o", "--out_path", default="outputs", type=str)
    parser.add_argument("-k", "--optimize_every_k", default=50, type=int, help="how often to re-optimize")
    parser.add_argument("-c", "--car_weight", default=1, type=float, help="weighting of cars in objective function")
    parser.add_argument(
        "-v", "--valid_edges_k", default=0, type=int, help="if subsampling edges, the number of nodes around SP"
    )
    parser.add_argument(
        "-p", "--penalty_shared", default=2, type=int, help="penalty factor for driving on a car lane by bike"
    )
    parser.add_argument(
        "-s", "--sp_method", default="od", type=str, help="Compute the shortest path either 'all_pairs' or 'od'"
    )
    parser.add_argument("--save_graph", action="store_true", help="if true, only creating one graph and saving it")
    parser.add_argument(
        "-a",
        "--algorithm",
        type=str,
        default="optimize",
        help="One of optimize, betweenness_topdown, betweenness_cartime, betweenness_biketime",
    )
    args = parser.parse_args()

    instance_name = args.instance
    path = os.path.join(args.data_path, instance_name)
    shared_lane_factor = args.penalty_shared  # how much to penalize biking on car lanes
    out_path = os.path.join(args.out_path, args.instance)
    sp_method = args.sp_method
    algorithm = args.algorithm
    assert algorithm in ["optimize", "betweenness_topdown", "betweenness_cartime", "betweenness_biketime"]
    out_path_ending = "_od" if sp_method == "od" else ""
    os.makedirs(out_path, exist_ok=True)

    np.random.seed(42)  # random seed for extending the od matrix
    # generate lane graph with snman
    G_lane = generate_motorized_lane_graph(
        os.path.join(path, "edges_all_attributes.gpkg"), os.path.join(path, "nodes_all_attributes.gpkg")
    )

    # load OD
    od = pd.read_csv(os.path.join(path, "od_matrix.csv"))
    od = od[od["s"] != od["t"]]
    # reduce OD matrix to nodes that are in G_lane
    node_list = list(G_lane.nodes())
    od = od[(od["s"].isin(node_list)) & (od["t"].isin(node_list))]

    # # Code to output instance properties
    # print(args.data_path.split(os.sep)[-1], ",", G_lane.number_of_nodes(), ",", G_lane.number_of_edges(), ",", len(od))
    # exit()

    assert nx.is_strongly_connected(G_lane), "G not connected"

    if "betweenness" in algorithm:
        print(f"Running betweenness algorithm {algorithm}")
        # get algorithm method
        algorithm_func, kwargs = algorithm_dict[algorithm]

        # run betweenness centrality algorithm for comparison
        pareto_between = algorithm_func(
            G_lane.copy(), sp_method=sp_method, od_matrix=od, weight_od_flow=WEIGHT_OD_FLOW, **kwargs
        )
        pareto_between.to_csv(os.path.join(out_path, f"real_pareto_{algorithm}{out_path_ending}.csv"), index=False)
        exit()

    # from here on, algorithm is "optimize"
    assert algorithm == "optimize"

    # extend OD matrix because otherwise we get disconnected car graph
    od = extend_od_circular(od, node_list)

    G_street = lane_to_street_graph(G_lane)

    # tune the car_weight
    runtimes_pareto = []
    for car_weight in [float(args.car_weight)]:  # [0.1, 0.25, 0.5, 1, 2, 4, 8]:
        print(f"Running LP for pareto frontier (car weight={car_weight})...")
        if ROUNDING_METHOD == "round_simple":
            tic = time.time()
            ip = define_IP(
                G_street,
                cap_factor=1,
                od_df=od,
                bike_flow_constant=FLOW_CONSTANT,
                car_flow_constant=FLOW_CONSTANT,
                shared_lane_factor=shared_lane_factor,
                weight_od_flow=WEIGHT_OD_FLOW,
                car_weight=car_weight,
                valid_edges_k=args.valid_edges_k,
            )
            toc = time.time()
            ip.optimize()
            toc2 = time.time()
            print("Time optimize", time.time() - tic)

            # nx.write_gpickle(G, "outputs/real_G.gpickle")
            capacity_values = output_to_dataframe(ip, G_street)
            # capacity_values.to_csv(os.path.join(out_path, f"real_capacities_{car_weight}.csv"), index=False)
            pareto_df = pareto_frontier(
                G_lane,
                capacity_values,
                shared_lane_factor=shared_lane_factor,
                sp_method=sp_method,
                od_matrix=od,
                weight_od_flow=WEIGHT_OD_FLOW,
                valid_edges_k=args.valid_edges_k,
            )

            if args.save_graph:
                # set desired number of bike ed
                num_bike_edges = int(RATIO_BIKE_EDGES * G_lane.number_of_edges())
                bike_G, car_G = rounding_and_splitting(capacity_values, bike_edges_to_add=num_bike_edges)
                G_lane_output = output_lane_graph(G_lane, bike_G, car_G, shared_lane_factor)
                edge_df = nx.to_pandas_edgelist(G_lane_output, edge_key="edge_key")
                edge_df.to_csv(os.path.join(out_path, "graph_edges.csv"), index=False)
                exit()
            del ip
        elif ROUNDING_METHOD == "round_bike_optimize":
            # compute the paretor frontier
            tic = time.time()

            opt = ParetoRoundOptimize(
                G_lane.copy(),
                od.copy(),
                optimize_every_x=args.optimize_every_k,
                car_weight=car_weight,
                sp_method=sp_method,
                shared_lane_factor=shared_lane_factor,
                weight_od_flow=WEIGHT_OD_FLOW,
                valid_edges_k=args.valid_edges_k,
            )
            if args.save_graph:
                num_bike_edges = int(RATIO_BIKE_EDGES * G_lane.number_of_edges())
                G_lane_output = opt.pareto(return_graph=True, max_bike_edges=num_bike_edges)
                edge_df = nx.to_pandas_edgelist(G_lane_output, edge_key="edge_key")
                edge_df.to_csv(os.path.join(out_path, "graph_edges.csv"), index=False)
                exit()
            else:
                pareto_df = opt.pareto()

            print("Time pareto", time.time() - tic)
            runtime_dict_from_pareto = opt.runtimes
            runtime_dict_from_pareto["time_pareto"] = time.time() - tic
            runtimes_pareto.append(runtime_dict_from_pareto)
        else:
            raise ValueError("Rounding method must be one of {round_bike_optimize, round_simple}")

        # save to file
        pareto_df.to_csv(
            os.path.join(out_path, f"real_pareto_optimize{out_path_ending}_{car_weight}_{args.optimize_every_k}.csv"),
            index=False,
        )

    # save runtimes
    with open(os.path.join(out_path, f"runtime_pareto_{car_weight}_{args.optimize_every_k}.json"), "w") as outfile:
        json.dump(runtimes_pareto, outfile)

    # combine all pareto frontiers
    combined_pareto = combine_pareto_frontiers(combine_paretos_from_path(out_path))
    combined_pareto.to_csv(os.path.join(out_path, f"real_pareto_combined_optimize{out_path_ending}.csv"), index=False)
