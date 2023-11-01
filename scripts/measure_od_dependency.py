import os
import time
import numpy as np
import networkx as nx
import pandas as pd
from ebike_city_tools.random_graph import random_lane_graph
from ebike_city_tools.utils import lane_to_street_graph, extend_od_circular, flow_to_df
from ebike_city_tools.optimize.linear_program import define_IP
from ebike_city_tools.optimize.round_simple import *
from ebike_city_tools.metrics import sp_length

OUT_PATH = "outputs"
os.makedirs(OUT_PATH, exist_ok=True)

SHARED_LANE_FACTOR = 2  # factor how much more expensive it is to bike on a car lane
nr_trials_per_graph = 3
CAR_WEIGHT = 1

res_df = []
# create graphs with the following number of nodes
for graph_num, n in enumerate([20, 20, 20, 20, 30, 30, 30, 30, 40, 40, 40, 40]):
    # create graph
    G_lane = random_lane_graph(n)
    G_base = lane_to_street_graph(G_lane)

    # full od matrix
    od_full = pd.DataFrame(np.array([[i, j] for i in range(n) for j in range(n)]), columns=["s", "t"])
    od_full["trips_per_day"] = 1

    # run several times to remove different parts from the OD matrix
    for i, trial in enumerate(range(nr_trials_per_graph)):
        # check for a reduction factor of 0.1, ... , 1
        for od_reduction in np.arange(1, 0, -0.1):
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
                    {"u_b(e)": ip.vars[f"u_{i},b"].x, "u_c(e)": ip.vars[f"u_{i},c"].x, "u": e[0], "v": e[1]}
                )
            fixed_values = pd.DataFrame(fixed_values).set_index(["u", "v"])

            # solve problem again with full od but fixed capacities - basically solving all-pairs min-cost flow problem
            ip_full = define_IP(
                G,
                cap_factor=1,
                od_df=od_full,
                shared_lane_factor=SHARED_LANE_FACTOR,
                car_weight=CAR_WEIGHT,
                fixed_values=fixed_values,
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
        pd.DataFrame(res_df).to_csv(os.path.join(OUT_PATH, "od_dependency_new.csv"))


def deprecated_version():
    """Method only kept here for reference"""
    if True:
        # get flow variables
        flow_df = flow_to_df(ip)
        flow_df = flow_df[flow_df["flow"] > 0]

        # # writing and reading for debugging
        # flow_df.to_csv("outputs/test_flow_solution.csv", index=False)
        # nx.write_gpickle(G, "outputs/test_G.gpickle")
        # flow_df = pd.read_csv("bike_lane_optimization/outputs/test_flow_solution.csv")
        # G = nx.read_gpickle("bike_lane_optimization/outputs/test_G.gpickle")

        # get car flows
        car_flows = flow_df[flow_df["var_type"] == "c"].groupby(["edge_u", "edge_v"])["flow"].max().to_dict()
        flow_edges = [tuple(t) for t in flow_df[["edge_u", "edge_v"]].values]
        # if edges are neither car nor bike lane (not needed due to reduced size of OD matrix, assign them to cars)
        for e in list(G.edges):
            if e not in flow_edges:
                car_flows[e] = 1

        # get bike lanes
        bike_flows = flow_df[flow_df["var_type"] == "b"].groupby(["edge_u", "edge_v"])["flow"].max().to_dict()

        # set up graph to compute all pairs shortest paths
        existing_car_edges = car_flows.keys()
        existing_bike_edges = bike_flows.keys()

        weight_sp_dict = {}
        for u, v, data in G.edges(data=True):
            inner_dict = {}
            if (u, v) in existing_car_edges:
                inner_dict["car_sp_weight"] = data["cartime"] / car_flows[(u, v)]
            else:
                inner_dict["car_sp_weight"] = np.inf
            if (u, v) in existing_bike_edges and bike_flows[(u, v)] > 1 / SHARED_LANE_FACTOR:
                inner_dict["bike_sp_weight"] = data["biketime"] / bike_flows[(u, v)]
            else:
                # treat as a shared edge
                inner_dict["bike_sp_weight"] = data["biketime"] * SHARED_LANE_FACTOR
            weight_sp_dict[(u, v)] = inner_dict
        #     print(u, v, data["biketime"])
        nx.set_edge_attributes(G, weight_sp_dict)

        # compute all pairs shortest paths
        car_sp = sp_length(G, "car_sp_weight", return_matrix=False)
        bike_sp = sp_length(G, "bike_sp_weight", return_matrix=False)
        assert car_sp < np.inf
