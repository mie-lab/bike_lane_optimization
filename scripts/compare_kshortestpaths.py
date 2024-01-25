import os
import time
import argparse
import numpy as np
import pandas as pd

from ebike_city_tools.synthetic import random_lane_graph, make_fake_od
from ebike_city_tools.od_utils import extend_od_circular
from ebike_city_tools.graph_utils import lane_to_street_graph
from ebike_city_tools.optimize.linear_program import define_IP


parser = argparse.ArgumentParser()
parser.add_argument("-o", "--out_path", default="outputs", type=str)
args = parser.parse_args()

OUT_PATH = args.out_path
os.makedirs(OUT_PATH, exist_ok=True)

SHARED_LANE_FACTOR = 2  # factor how much more expensive it is to bike on a car lane
NUMBER_PATHS_LIST = [0, 20, 30, 40, 50]  # this is actually the number of nodes used aroudn the shortest path
# [0, 1, 2, 3, 4, 5, 6] for method "determine_valid_arcs"
CAR_WEIGHT = 1
OD_REDUCTION = 0.01
graph_trial_size_list = [100, 100, 100, 250, 250, 250, 200, 200, 200, 150, 150, 150]

np.random.seed(42)

res_df = []
# create graphs with the following number of nodes
for graph_num, n in enumerate(graph_trial_size_list):
    # create graph
    G_lane = random_lane_graph(n)
    G_base = lane_to_street_graph(G_lane)
    od = make_fake_od(n, int(OD_REDUCTION * n**2), nodes=G_lane.nodes)
    od = extend_od_circular(od, list(G_lane.nodes()))

    # use different k shortest paths parameter
    for number_paths in NUMBER_PATHS_LIST:
        G = G_base.copy()

        # # for logging the average / sum of the number of paths per OD pair
        # if number_paths > 0:
        #     number_valid_arcs = [len(arcs) for k, arcs in valid_arcs.items()]
        # else:
        #     number_valid_arcs = [G.number_of_edges() for _ in range(len(od))]

        # initialize and optimize with this od matrix
        tic = time.time()
        # this is the main change -> use valid_edges_k argument
        ip = define_IP(
            G,
            cap_factor=1,
            od_df=od,
            valid_edges_k=number_paths,
            shared_lane_factor=SHARED_LANE_FACTOR,
            car_weight=CAR_WEIGHT,
        )
        toc = time.time()
        ip.optimize()
        toc2 = time.time()
        obj_value = ip.objective_value

        # get capacities -> they are fixed from now on
        fixed_values = []
        for i, e in enumerate(G.edges):
            fixed_values.append({"u_b(e)": ip.vars[f"u_{e},b"].x, "u_c(e)": ip.vars[f"u_{e},c"].x, "Edge": e})
        fixed_values = pd.DataFrame(fixed_values)  # .set_index(["u", "v"])

        # solve problem again with full od but fixed capacities
        # this way, the capacities are set to the values optimized with the reduced edge set, but the flow can be
        # divided differently - this time to all edges. We therefore compute the objective value when the capacities are
        # set by the simplified algorithm, but the cars can still go where it's best and not just along the edge set
        ip_full = define_IP(
            G,
            cap_factor=1,
            od_df=od,
            shared_lane_factor=SHARED_LANE_FACTOR,
            car_weight=CAR_WEIGHT,
            fixed_edges=fixed_values,
        )
        ip_full.optimize()
        new_obj_value = ip_full.objective_value

        # store results
        res_dict = {
            "instance": graph_num,
            "objective_k": obj_value,
            "objective_all_paths": new_obj_value,
            "number_nodes": n,
            "number_edges": G_base.number_of_edges(),
            "k_shortest_paths": number_paths,
            "od_pairs": len(od),
            # "sum_valid_arcs": sum(number_valid_arcs),
            # "mean_valid_arcs": np.mean(number_valid_arcs),
            "runtime_init": toc - tic,
            "runtime_optim": toc2 - toc,
        }
        res_df.append(res_dict)
        print("\n-------------")
        print(res_df[-1])
        print("\n-------------")

    # save in every intermediate step in case something throws an error
    pd.DataFrame(res_df).to_csv(os.path.join(OUT_PATH, "k_shortest_path_big.csv"), index=False)
