import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ebike_city_tools.synthetic import random_lane_graph, make_fake_od
from ebike_city_tools.optimize.round_optimized import ParetoRoundOptimize
from ebike_city_tools.iterative_algorithms import topdown_betweenness_pareto, betweenness_pareto

CAR_WEIGHT = 1
WEIGHT_OD_FLOW = False
SHARED_LANE_FACTOR = 2
OD_REDUCTION = 0.4
SP_METHOD = "od"
kwargs = {"weight_od_flow": WEIGHT_OD_FLOW, "sp_method": SP_METHOD}
algorithm_dict = {
    "betweenness_topdown": (topdown_betweenness_pareto, kwargs),
    "betweenness_cartime": (betweenness_pareto, {"betweenness_attr": "car_time", **kwargs}),
    "betweenness_biketime": (betweenness_pareto, {"betweenness_attr": "bike_time", **kwargs}),
}
OUT_PATH = "outputs/round_optimized"
os.makedirs(OUT_PATH, exist_ok=True)

OPTIMIZE_EVERY_LIST = [100, 25, 15, 10, 5]
graph_trial_size_list = [20, 20, 20, 20, 30, 30, 30, 30, 40, 40, 40, 40, 50, 50, 50, 50, 60, 60, 60, 60]

np.random.seed(42)

if __name__ == "__main__":
    for graph_trial, graph_size in enumerate(graph_trial_size_list):
        G_lane = random_lane_graph(graph_size)
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
        for optimize_every in OPTIMIZE_EVERY_LIST:
            opt = ParetoRoundOptimize(
                G_lane.copy(), od.copy(), optimize_every_x=optimize_every, car_weight=CAR_WEIGHT, **kwargs
            )
            pareto_front = opt.pareto()
            pareto_front.to_csv(
                os.path.join(OUT_PATH, f"pareto_optimize{CAR_WEIGHT}_{graph_trial}_{optimize_every}.csv")
            )

    # -------- PLOTTING -------------
    # this code loads all the result files and plots them
    # set the color maps
    colors = plt.cm.viridis(np.linspace(0, 1, len(OPTIMIZE_EVERY_LIST)))
    cols_betweenness = ["black", "grey", "lightgrey"]
    # load data and plot for every graph trial
    for graph_trial in range(graph_trial_size_list):
        plt.figure(figsize=(8, 5))
        for baseline, col in zip(algorithm_dict.keys(), cols_betweenness):
            pareto_front = pd.read_csv(os.path.join(OUT_PATH, f"pareto_{baseline}_{graph_trial}.csv"))
            plt.plot(pareto_front["bike_time"], pareto_front["car_time"], label=baseline, c=col, linestyle="--")

        for i, optimize_every in enumerate(OPTIMIZE_EVERY_LIST):
            pareto_front = pd.read_csv(
                os.path.join(OUT_PATH, f"pareto_optimize{CAR_WEIGHT}_{graph_trial}_{optimize_every}.csv")
            )
            plt.plot(pareto_front["bike_time"], pareto_front["car_time"], label=optimize_every, c=colors[i])

        plt.xlabel("bike travel time")
        plt.ylabel("car travel time")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_PATH, f"figures_trial_{graph_trial}.pdf"))
