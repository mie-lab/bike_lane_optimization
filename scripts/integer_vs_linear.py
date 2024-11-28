import os
import time
import numpy as np
import networkx as nx
import pandas as pd
from ebike_city_tools.synthetic import random_lane_graph, make_fake_od
from ebike_city_tools.metrics import compute_travel_times
from ebike_city_tools.optimize.optimizer import Optimizer
from ebike_city_tools.od_utils import extend_od_circular
from ebike_city_tools.graph_utils import lane_to_street_graph
from ebike_city_tools.optimize.round_simple import graph_from_integer_solution
from ebike_city_tools.optimize.round_optimized import ParetoRoundOptimize

OUT_PATH = "outputs"
os.makedirs(OUT_PATH, exist_ok=True)
os.makedirs("outputs/int_vs_lin_graphs", exist_ok=True)
NR_ITERS = 15
OPTIMIZE_EVERY_K = 10
shared_lane_factor = 2
SP_METHOD = "od"
WEIGHT_OD_FLOW = False
od_factor = 0.01

if __name__ == "__main__":
    np.random.seed(42)
    res_df = []
    for i in range(NR_ITERS):
        # only run on graph of size 30 to make it feasible
        for size in [80, 60]:
            neighbor_p = np.random.choice(["[0.6, 0.3, 0.1]", "[0.7, 0.2, 0.1]", "[0.8, 0.1, 0.1]"])
            G_lane = random_lane_graph(size, neighbor_p=eval(neighbor_p))
            nx.write_adjlist(G_lane, f"outputs/int_vs_lin_graphs/graph_{i}_{size}.gpickle")

            centralities = list(nx.edge_betweenness_centrality(G_lane).values())
            bet_mean, bet_std = np.mean(centralities), np.std(centralities)

            cluster = list(nx.clustering(nx.DiGraph(G_lane)).values())
            cluster_mean, cluster_std = np.mean(cluster), np.std(cluster)
            print("----------")
            print("Graph", G_lane.number_of_nodes(), G_lane.number_of_edges())
            print("betweenness und clustering", bet_mean, bet_std, cluster_mean, cluster_std, neighbor_p)

            G = lane_to_street_graph(G_lane)
            # test for graphs with OD matrix of 10% of the all-pairs SP
            # define graph
            od = make_fake_od(size, int(od_factor * size**2), nodes=G.nodes)
            od = extend_od_circular(od, list(G_lane.nodes()))

            for OPTIMIZE_EVERY_K in [10, 20]:

                # run for several car weights to get a pareto frontier also for the integer problem
                for car_weight in [0.5, 0.75, 1, 2, 4, 8]:
                    for integer_problem, name in zip([True, False], ["integer", "linear"]):
                        # transform the graph layout into travel times, including gradient and penalty factor for using
                        if integer_problem:
                            if OPTIMIZE_EVERY_K != 10:
                                # need to do linear problem only ones
                                continue
                            optim = Optimizer(
                                graph=G.copy(),
                                od_matrix=od,
                                integer_problem=integer_problem,
                                shared_lane_factor=shared_lane_factor,  # factor how much worse it is to bike on a car lane
                                car_weight=car_weight,
                                weight_od_flow=WEIGHT_OD_FLOW,
                            )
                            tic = time.time()
                            optim.init_lp()
                            toc = time.time()
                            obj_value = optim.optimize()
                            toc_finished = time.time()

                            capacity_values = optim.get_solution()
                            # # first check whether car graph is strongly connected
                            # car_G_init = ceiled_car_graph(capacity_values.copy())
                            # if not nx.is_strongly_connected(car_G_init):
                            #     continue

                            bike_G, car_G = graph_from_integer_solution(capacity_values)
                            res_dict_list = [
                                compute_travel_times(
                                    G_lane,
                                    bike_G,
                                    car_G,
                                    od_matrix=od,
                                    sp_method=SP_METHOD,
                                    shared_lane_factor=shared_lane_factor,
                                    weight_od_flow=WEIGHT_OD_FLOW,
                                )
                            ]
                        else:
                            # for linear, we have to compute the paretor frontier
                            opt = ParetoRoundOptimize(
                                G_lane.copy(),
                                od.copy(),
                                optimize_every_x=OPTIMIZE_EVERY_K,
                                car_weight=car_weight,
                                sp_method=SP_METHOD,
                                shared_lane_factor=shared_lane_factor,
                                weight_od_flow=WEIGHT_OD_FLOW,
                            )
                            res_dict_list = opt.pareto(return_list=True)
                        # add general infos to integer or linear (pareto) solution
                        for r in res_dict_list:
                            r.update(
                                {
                                    "iter": i,
                                    "name": name,
                                    "nodes": size,
                                    "edges": G_lane.number_of_edges(),
                                    "betweenness_mean": bet_mean,
                                    "betweenness_std": bet_std,
                                    "cluster_mean": cluster_mean,
                                    "cluster_std": cluster_std,
                                    "od_size": len(od),
                                    "car_weight": car_weight,
                                    "opt_value": obj_value,
                                    "optimize_every": OPTIMIZE_EVERY_K,
                                    "time init": toc - tic,
                                    "time optimize": toc_finished - toc,
                                }
                            )
                        res_df.extend(res_dict_list)
                        print("----------")
                        print(res_df[-1])
                        print("----------")
                # save updated df in every iteration
                pd.DataFrame(res_df).to_csv(os.path.join(OUT_PATH, "integer_vs_linear.csv"), index=False)
