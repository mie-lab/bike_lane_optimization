import os
import numpy as np
from ebike_city_tools.synthetic import random_lane_graph, make_fake_od
from ebike_city_tools.optimize.round_optimized import ParetoRoundOptimize
from ebike_city_tools.iterative_algorithms import topdown_betweenness_pareto, betweenness_pareto

GRAPH_SIZE = 30
CAR_WEIGHT = 1
WEIGHT_OD_FLOW = False
SHARED_LANE_FACTOR = 2
SP_METHOD = "od"
kwargs = {"weight_od_flow": WEIGHT_OD_FLOW, "sp_method": SP_METHOD}
algorithm_dict = {
    "betweenness_topdown": (topdown_betweenness_pareto, {}),
    "betweenness_cartime": (betweenness_pareto, {"betweenness_attr": "car_time"}),
    "betweenness_biketime": (betweenness_pareto, {"betweenness_attr": "bike_time"}),
}
OUT_PATH = "outputs/round_optimized"
os.makedirs(OUT_PATH, exist_ok=True)

np.random.seed(42)

if __name__ == "__main__":
    for graph_trial in range(5):
        G_lane = random_lane_graph(GRAPH_SIZE)
        od = make_fake_od(GRAPH_SIZE, 90, nodes=G_lane.nodes)

        # Run all baselines on this graph
        for algorithm in algorithm_dict.keys():
            # get algorithm method
            algorithm_func, kwargs_added = algorithm_dict[algorithm]
            kwargs.update(kwargs_added)
            # run betweenness centrality algorithm for comparison
            pareto_between = algorithm_func(G_lane.copy(), od_matrix=od.copy(), **kwargs)
            pareto_between.to_csv(os.path.join(OUT_PATH, f"pareto_{algorithm}_{graph_trial}.csv"), index=False)

        # TODO: run alternative algorithm with car and bike edge rounding

        # Run ParetoRoundOptimize with varying batch size
        for optimize_every in [100, 25, 15, 10, 5, 2]:
            opt = ParetoRoundOptimize(
                G_lane.copy(), od.copy(), optimize_every_x=optimize_every, car_weight=CAR_WEIGHT, **kwargs
            )
            pareto_front = opt.pareto()
            pareto_front.to_csv(
                os.path.join(OUT_PATH, f"pareto_optimize{CAR_WEIGHT}_{graph_trial}_{optimize_every}.csv")
            )
