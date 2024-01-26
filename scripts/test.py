import os
import time
import matplotlib.pyplot as plt
import numpy as np
from ebike_city_tools.synthetic import random_lane_graph, make_fake_od
from ebike_city_tools.optimize.round_optimized import ParetoRoundOptimize
from ebike_city_tools.graph_utils import lane_to_street_graph
from ebike_city_tools.iterative_algorithms import betweenness_pareto, topdown_betweenness_pareto

OUT_PATH = "outputs"
os.makedirs(OUT_PATH, exist_ok=True)

shared_lane_factor = 2

if __name__ == "__main__":
    np.random.seed(100)

    G_lane = random_lane_graph(10)
    G_street = lane_to_street_graph(G_lane)
    od = make_fake_od(10, 50, nodes=G_street.nodes)

    # # for flow-sum-constraint: compute maximal number of paths that could pass through one edge
    # max_paths_one_edge = len(od)  # maximum number of paths through one edge corresponds to number of OD-pairs
    # FACTOR_MAX_PATHS = 0.5  # only half of the paths are allowed to use the same street
    # cap_factor = max_paths_one_edge * FACTOR_MAX_PATHS
    opt = ParetoRoundOptimize(
        G_lane.copy(),
        od.copy(),
        optimize_every_x=100,
        car_weight=2,
        sp_method="od",
        only_double_bikelanes=True,
        shared_lane_factor=shared_lane_factor,
    )
    pareto_df = opt.pareto()
    # pareto_df = betweenness_pareto(G_lane.copy(), od.copy(), "od", shared_lane_factor=2)
    # pareto_df = topdown_betweenness_pareto(G_lane.copy(), od.copy(), "od", shared_lane_factor=2, fix_multilane=True)

    # save to csv
    # pareto_df.to_csv(os.path.join(OUT_PATH, "test_pareto_df.csv"))

    # plot pareto frontier
    plt.scatter(pareto_df["bike_time"], pareto_df["car_time"])
    # plt.savefig("comparison_front.png")
    plt.show()

    # save the new graph --> undirected
    # nx.write_gpickle(G, "outputs/test_G_random.gpickle")
