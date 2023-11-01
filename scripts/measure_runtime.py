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
NR_ITERS = 1

np.random.seed(1)

if __name__ == "__main__":
    res_df = []
    # test for graphs with 30-100 nodes
    for size in np.arange(100, 200, 10):
        G_lane = random_lane_graph(size)
        G = lane_to_street_graph(G_lane)
        # test for graphs with OD matrix of 2 times, 3 times, or 4 times as many entries as the number of nodes
        od_size_list = (np.random.rand(2) * 800 + 200).astype(int)
        for od_size in od_size_list:
            if 3 * od_size * G.number_of_edges() > 3000000 or od_size > size**2:
                continue

            # test for several iterations because randomized
            for i in range(NR_ITERS):
                od = make_fake_od(int(size), od_size, nodes=G.nodes)
                od = extend_od_circular(od, list(G.nodes()))

                tic = time.time()
                new_ip = define_IP(G, cap_factor=1, od_df=od)
                toc = time.time()
                new_ip.optimize()
                toc_finished = time.time()
                res_df.append(
                    {
                        "nodes": G.number_of_nodes(),
                        "edges": G.number_of_edges(),
                        "od_size": len(od),
                        "opt_value": new_ip.objective_value,
                        "time init": toc - tic,
                        "time optimize": toc_finished - toc,
                    }
                )
                print("----------")
                print(res_df[-1])
                print("----------")
            # save updated df in every iteration
            pd.DataFrame(res_df).to_csv(os.path.join(OUT_PATH, "runtime.csv"))
