import os
import time
import numpy as np
import networkx as nx
import pandas as pd
from ebike_city_tools.synthetic import random_lane_graph, make_fake_od
from ebike_city_tools.metrics import compute_travel_times
from ebike_city_tools.optimize.optimizer import Optimizer
from ebike_city_tools.utils import lane_to_street_graph, extend_od_circular
from ebike_city_tools.optimize.round_simple import ceiled_car_graph, pareto_frontier, graph_from_integer_solution
from ebike_city_tools.optimize.round_optimized import ParetoRoundOptimize

OUT_PATH = "outputs"
os.makedirs(OUT_PATH, exist_ok=True)
NR_ITERS = 50
OPTIMIZE_EVERY_K = 10
shared_lane_factor = 2
SP_METHOD = "od"
WEIGHT_OD_FLOW = False

if __name__ == "__main__":
    np.random.seed(42)
    res_df = []
    for i in range(NR_ITERS):
        # only run on graph of size 30 to make it feasible
        for size in [30]:
            G_lane = random_lane_graph(size)
            G = lane_to_street_graph(G_lane)
            # test for graphs with OD matrix of 10% of the all-pairs SP
            for od_factor in [0.1]:
                # define graph
                od = make_fake_od(size, od_factor * size**2, nodes=G.nodes)
                od = extend_od_circular(od, list(G_lane.nodes()))

                # run for several car weights to get a pareto frontier also for the integer problem
                for car_weight in [0.1, 0.5, 0.75, 1, 2, 4, 8]:
                    for integer_problem, name in zip([True, False], ["integer", "linear"]):
                            
                        # transform the graph layout into travel times, including gradient and penalty factor for using
                        if integer_problem:
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
                                return_list=True,
                                weight_od_flow=WEIGHT_OD_FLOW,
                            )
                            res_dict_list = opt.pareto()
                        # add general infos to integer or linear (pareto) solution
                        for r in res_dict_list:
                            r.update(
                                {
                                    "iter": i,
                                    "name": name,
                                    "nodes": size,
                                    "edges": G.number_of_edges(),
                                    "od_size": len(od),
                                    "car_weight": car_weight,
                                    "opt_value": obj_value,
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
