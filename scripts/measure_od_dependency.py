import os
import time
import numpy as np
import networkx as nx
import pandas as pd
from ebike_city_tools.synthetic import random_lane_graph
from ebike_city_tools.utils import lane_to_street_graph, extend_od_circular
from ebike_city_tools.optimize.linear_program import define_IP

OUT_PATH = "outputs"
os.makedirs(OUT_PATH, exist_ok=True)

SHARED_LANE_FACTOR = 2  # factor how much more expensive it is to bike on a car lane
nr_trials_per_graph = 3
CAR_WEIGHT = 1
graph_trial_size_list = [20, 20, 20, 20, 30, 30, 30, 30, 40, 40, 40, 40, 50, 50, 50, 50, 60, 60, 60, 60]

np.random.seed(42)

res_df = []
# create graphs with the following number of nodes
for graph_num, n in enumerate(graph_trial_size_list):
    # create graph
    G_lane = random_lane_graph(n)
    G_base = lane_to_street_graph(G_lane)

    # full od matrix
    od_full = pd.DataFrame(np.array([[i, j] for i in range(n) for j in range(n)]), columns=["s", "t"])
    od_full["trips_per_day"] = 1

    # run several times to remove different parts from the OD matrix
    for i, trial in enumerate(range(nr_trials_per_graph)):
        # check for a reduction factor of 0.1, ... , 1
        for od_reduction in list(np.arange(1, 0, -0.1)) + [0.05]:
            G = G_base.copy()

            # reduce od matrix
            new_od_size = int(len(od_full) * od_reduction)
            od = od_full.sample(new_od_size, replace=False)
            # extend OD matrix because otherwise we get disconnected graphs
            od = extend_od_circular(od, list(G_lane.nodes()))

            # initialize and optimize with this od matrix
            tic = time.time()
            ip = define_IP(G, cap_factor=1, od_df=od, shared_lane_factor=SHARED_LANE_FACTOR, car_weight=CAR_WEIGHT)
            toc = time.time()
            ip.optimize()
            toc2 = time.time()
            obj_value = ip.objective_value

            # get capacities -> they are fixed from now on
            fixed_values = []
            for i, e in enumerate(G.edges):
                fixed_values.append(
                    {"u_b(e)": ip.vars[f"u_{e},b"].x, "u_c(e)": ip.vars[f"u_{e},c"].x, "Edge": e}
                )
            fixed_values = pd.DataFrame(fixed_values) #.set_index(["u", "v"])

            # solve problem again with full od but fixed capacities - basically solving all-pairs min-cost flow problem
            ip_full = define_IP(
                G,
                cap_factor=1,
                od_df=od_full,
                shared_lane_factor=SHARED_LANE_FACTOR,
                car_weight=CAR_WEIGHT,
                fixed_edges=fixed_values,
            )
            ip_full.optimize()
            new_obj_value = ip_full.objective_value

            # store results
            res_dict = {
                "instance": graph_num,
                "objective_od": obj_value,
                "objective_all_pairs": new_obj_value,
                "number_nodes": n,
                "number_edges": G_base.number_of_edges(),
                "od_ratio": od_reduction,
                "od_pairs": len(od),
                "nr_trial": trial,
                "runtime_init": toc - tic,
                "runtime_optim": toc2 - toc,
            }
            res_df.append(res_dict)
            print("\n-------------")
            print(res_df[-1])
            print("\n-------------")

        # save in every intermediate step in case something throws an error
        pd.DataFrame(res_df).to_csv(os.path.join(OUT_PATH, "od_dependency.csv"))
