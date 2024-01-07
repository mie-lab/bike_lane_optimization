import os
import time
import numpy as np
import networkx as nx
import pandas as pd
from ebike_city_tools.synthetic import random_lane_graph, make_fake_od
from ebike_city_tools.utils import lane_to_street_graph, extend_od_circular
from ebike_city_tools.optimize.linear_program import define_IP
from ebike_city_tools.optimize.round_optimized_sort_selection import determine_valid_arcs, valid_arcs_spatial_selection

OUT_PATH = "outputs"
os.makedirs(OUT_PATH, exist_ok=True)

SHARED_LANE_FACTOR = 2  # factor how much more expensive it is to bike on a car lane
NUMBER_PATHS_LIST = [0, 2, 4, 8, 16, 24]
CAR_WEIGHT = 1
OD_REDUCTION = 0.1
graph_trial_size_list = [30, 30, 30, 30, 40, 40, 40, 40, 50, 50, 50, 50, 60, 60, 60, 60]

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
    # for number_paths in NUMBER_PATHS_LIST:
    for number_paths in NUMBER_PATHS_LIST:
        G = G_base.copy()

        if number_paths > 0:
            valid_arcs = valid_arcs_spatial_selection(od, G, number_paths)
            # valid_arcs = determine_valid_arcs(od, G, number_paths)
            number_valid_arcs = [len(arcs) for k, arcs in valid_arcs.items()]
            print(number_paths, sum(number_valid_arcs), G.number_of_edges() * len(od))
        else:
            valid_arcs = None
            number_valid_arcs = [G.number_of_edges() for _ in range(len(od))]

        # initialize and optimize with this od matrix
        tic = time.time()
        ip = define_IP(
            G,
            cap_factor=1,
            od_df=od,
            valid_edges_per_od_pair=valid_arcs,  # this is the main change -> only use valid arcs
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

        # solve problem again with full od but fixed capacities - basically solving all-pairs min-cost flow problem
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
            "sum_valid_arcs": sum(number_valid_arcs),
            "mean_valid_arcs": np.mean(number_valid_arcs),
            "runtime_init": toc - tic,
            "runtime_optim": toc2 - toc,
        }
        res_df.append(res_dict)
        print("\n-------------")
        print(res_df[-1])
        print("\n-------------")

    # save in every intermediate step in case something throws an error
    pd.DataFrame(res_df).to_csv(os.path.join(OUT_PATH, "k_shortest_path_comparison.csv"), index=False)
