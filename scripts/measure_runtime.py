import os
import numpy as np
import time
import pandas as pd

from ebike_city_tools.optimize.linear_program import define_IP
from ebike_city_tools.optimize.utils import output_to_dataframe, make_fake_od


from ebike_city_tools.random_graph import city_graph, lane_to_street_graph

OUT_PATH = "outputs"
os.makedirs(OUT_PATH, exist_ok=True)
NR_ITERS = 5

if __name__ == "__main__":
    res_df = []
    # test for graphs with 30-100 nodes
    for size in np.arange(30, 110, 10):
        G_city = city_graph(size)
        G = lane_to_street_graph(G_city)
        # test for graphs with OD matrix of 2 times, 3 times, or 4 times as many entries as the number of nodes
        for od_factor in [2, 3, 4]:
            # test for several iterations because randomized
            for i in range(NR_ITERS):
                od = make_fake_od(size, od_factor * size, nodes=G.nodes)

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
