import os
import time
import matplotlib.pyplot as plt
import numpy as np
from ebike_city_tools.optimize.optimizer import Optimizer
from ebike_city_tools.optimize.utils import output_to_dataframe, flow_to_df
from ebike_city_tools.random_graph import random_lane_graph
from ebike_city_tools.optimize.utils import make_fake_od
from ebike_city_tools.optimize.round_simple import pareto_frontier
from ebike_city_tools.utils import lane_to_street_graph

OUT_PATH = "outputs"
os.makedirs(OUT_PATH, exist_ok=True)

shared_lane_factor = 2

if __name__ == "__main__":
    np.random.seed(20)

    G_lane = random_lane_graph(30)
    G_street = lane_to_street_graph(G_lane)

    od = make_fake_od(30, 90, nodes=G_street.nodes)
    # # for flow-sum-constraint: compute maximal number of paths that could pass through one edge
    # max_paths_one_edge = len(od)  # maximum number of paths through one edge corresponds to number of OD-pairs
    # FACTOR_MAX_PATHS = 0.5  # only half of the paths are allowed to use the same street
    # cap_factor = max_paths_one_edge * FACTOR_MAX_PATHS

    optim = Optimizer(graph=G_street, od_matrix=od, car_weight=5, shared_lane_factor=shared_lane_factor)
    optim.init_lp()
    obj_value = optim.optimize()
    capacity_values, flow_df = optim.get_solution(return_flow=True)

    flow_df.to_csv(os.path.join(OUT_PATH, "test_flow_solution.csv"), index=False)
    capacity_values.to_csv(os.path.join(OUT_PATH, "test_lp_solution.csv"), index=False)

    # for linear, we have to compute the paretor frontier
    pareto_df = pareto_frontier(G_lane, capacity_values, shared_lane_factor=shared_lane_factor)
    pareto_df.to_csv(os.path.join(OUT_PATH, "test_pareto_df.csv"))
    # plot pareto frontier
    # plt.scatter(pareto_df["bike_time"], pareto_df["car_time"])
    # plt.show()

    print("OPT VALUE", obj_value)
    # save the new graph --> undirected
    # nx.write_gpickle(G, "outputs/test_G_random.gpickle")
