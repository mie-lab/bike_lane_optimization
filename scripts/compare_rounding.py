from ebike_city_tools.random_graph import random_lane_graph, make_fake_od
from ebike_city_tools.optimize.round_optimized import ParetoRoundOptimize

if __name__ == "__main__":
    for graph_trial in range(5):
        G_lane = random_lane_graph(30)
        od = make_fake_od(30, 90, nodes=G_lane.nodes)
        for optimize_every in [100, 25, 15, 10, 5, 2]:
            opt = ParetoRoundOptimize(G_lane.copy(), od, optimize_every_x=optimize_every, car_weight=1)
            pareto_front = opt.pareto()
            pareto_front.to_csv(f"outputs/round_optimized/pareto_{graph_trial}_{optimize_every}.csv")
