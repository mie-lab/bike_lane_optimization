import os
import numpy as np
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from ebike_city_tools.synthetic import random_lane_graph, make_fake_od
from ebike_city_tools.optimize.round_optimized import ParetoRoundOptimize
from ebike_city_tools.iterative_algorithms import topdown_betweenness_pareto, betweenness_pareto

WEIGHT_OD_FLOW = False
PLOTTING = False
SHARED_LANE_FACTOR = 2
OD_REDUCTION = 0.1
SP_METHOD = "od"
kwargs = {"weight_od_flow": WEIGHT_OD_FLOW, "sp_method": SP_METHOD}
algorithm_dict = {
    "betweenness_topdown": (topdown_betweenness_pareto, kwargs),
    "betweenness_cartime": (betweenness_pareto, {"betweenness_attr": "car_time", **kwargs}),
    "betweenness_biketime": (betweenness_pareto, {"betweenness_attr": "bike_time", **kwargs}),
}

OPTIMIZE_EVERY_LIST = [100, 50, 25, 10, 5]
CAR_WEIGHT_LIST = [0.1, 0.5, 1, 2, 4, 8]
graph_trial_size_list = [20, 20, 20, 20, 30, 30, 30, 30, 40, 40, 40, 40, 50, 50, 50, 50, 60, 60, 60, 60]

np.random.seed(42)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--out_path", default="outputs/compare_rounding", type=str)
    args = parser.parse_args()
    OUT_PATH = args.out_path
    os.makedirs(OUT_PATH, exist_ok=True)
    for graph_trial, graph_size in enumerate(graph_trial_size_list):
        G_lane = random_lane_graph(graph_size)
        print(G_lane.number_of_edges())
        od = make_fake_od(graph_size, int(OD_REDUCTION * graph_size**2), nodes=G_lane.nodes)

        # Run all baselines on this graph
        for algorithm in algorithm_dict.keys():
            # get algorithm method
            algorithm_func, kwargs_betweenness = algorithm_dict[algorithm]
            # run betweenness centrality algorithm for comparison
            pareto_between = algorithm_func(G_lane.copy(), od_matrix=od.copy(), **kwargs_betweenness)
            pareto_between.to_csv(os.path.join(OUT_PATH, f"pareto_{algorithm}_{graph_trial}.csv"), index=False)

        # TODO: run alternative algorithm with car and bike edge rounding

        # Run ParetoRoundOptimize with varying batch size
        for trial, optimize_every in enumerate([G_lane.number_of_edges() + 10] + OPTIMIZE_EVERY_LIST):
            for car_weight in CAR_WEIGHT_LIST:
                opt = ParetoRoundOptimize(
                    G_lane.copy(), od.copy(), optimize_every_x=optimize_every, car_weight=car_weight, **kwargs
                )
                pareto_front = opt.pareto()
                optimize_every_name = "none" if trial == 0 else optimize_every
                pareto_front.to_csv(
                    os.path.join(OUT_PATH, f"pareto_optimize{car_weight}_{graph_trial}_{optimize_every_name}.csv")
                )

    # -------- PLOTTING -------------
    # this code loads all the result files and plots them
    # set the color maps
    if PLOTTING:
        colors = plt.cm.viridis(np.linspace(0, 1, len(OPTIMIZE_EVERY_LIST)))
        cols_betweenness = ["black", "grey", "lightgrey"]
        # load data and plot for every graph trial
        for graph_trial in graph_trial_size_list:
            plt.figure(figsize=(8, 5))
            for baseline, col in zip(algorithm_dict.keys(), cols_betweenness):
                pareto_front = pd.read_csv(os.path.join(OUT_PATH, f"pareto_{baseline}_{graph_trial}.csv"))
                plt.plot(pareto_front["bike_time"], pareto_front["car_time"], label=baseline, c=col, linestyle="--")

            for i, optimize_every in enumerate(OPTIMIZE_EVERY_LIST):
                for car_weight in CAR_WEIGHT_LIST:
                    pareto_front = pd.read_csv(
                        os.path.join(OUT_PATH, f"pareto_optimize{car_weight}_{graph_trial}_{optimize_every}.csv")
                    )
                    plt.plot(
                        pareto_front["bike_time"],
                        pareto_front["car_time"],
                        label=f"{optimize_every}-{car_weight}",
                        c=colors[i],
                    )

            plt.xlabel("bike travel time")
            plt.ylabel("car travel time")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(OUT_PATH, f"figures_trial_{graph_trial}.pdf"))
