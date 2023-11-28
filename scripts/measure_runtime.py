import os
import numpy as np
import time
import pandas as pd

from ebike_city_tools.optimize.linear_program import define_IP
from ebike_city_tools.synthetic import make_fake_od
from ebike_city_tools.utils import lane_to_street_graph, extend_od_circular
from ebike_city_tools.synthetic import random_lane_graph

OUT_PATH = "outputs"
os.makedirs(OUT_PATH, exist_ok=True)
NR_ITERS = 2

np.random.seed(1)

if __name__ == "__main__":
    res_df = []
    for desired_variable in np.arange(100000, 3200000, 300000):
        for size in [100, 150, 200, 250, 300, 350, 400]:
            # test for several iterations because randomized
            for i in range(NR_ITERS):
                G_lane = random_lane_graph(size)
                G = lane_to_street_graph(G_lane)
                n, m = G.number_of_nodes(), G.number_of_edges()
                # solve for od factor
                od_factor = (desired_variable - 2 * m) / (n**2 * m * 3) - 1 / n
                od_size = od_factor * n**2
                if od_factor < 0.001 or od_factor > 1:
                    continue

                od = make_fake_od(int(size), int(round(od_size)), nodes=G.nodes)
                od = extend_od_circular(od, list(G.nodes()))
                print("planned od size", int(round(od_size)), "actual od size", len(od))

                tic = time.time()
                new_ip = define_IP(G, cap_factor=1, od_df=od)
                toc = time.time()
                new_ip.optimize()
                toc_finished = time.time()
                time_init = toc - tic
                time_optim = toc_finished - toc
                time_total = time_init + time_optim
                opt_value = new_ip.objective_value
                nr_variables_mio = (3 * (len(od) * G.number_of_edges()) + 2 * G.number_of_edges()) / 1000000
                print("Desired variable:", desired_variable, "Real:", nr_variables_mio)

                # # for testing
                # time_init = 0.0001 * nr_variables_mio
                # time_optim = 0.0001 * nr_variables_mio
                # opt_value = 1

                res_df.append(
                    {
                        "nodes": G.number_of_nodes(),
                        "edges": G.number_of_edges(),
                        "od_size": len(od),
                        "opt_value": opt_value,
                        "time_init": time_init,
                        "time_optim": time_optim,
                        "nr_variables": nr_variables_mio,
                        "desired_vars": desired_variable,
                    }
                )
                print("----------")
                print(res_df[-1])
                print("----------")
            # save updated df in every iteration
            pd.DataFrame(res_df).to_csv(os.path.join(OUT_PATH, "runtime.csv"), index=False)
