import time
import pandas as pd

from ebike_city_tools.optimize.linear_program import initialize_IP
from ebike_city_tools.optimize.utils import output_to_dataframe


from ebike_city_tools.random_graph import base_graph_with_capacity

if __name__ == "__main__":
    # test on 5 different graphs for each size
    res_df = []
    for size in [35, 50, 60]:
        for i in range(3):
            G = base_graph_with_capacity(size)
            tic = time.time()
            new_ip = initialize_IP(G)
            toc = time.time()
            new_ip.optimize()
            toc_finished = time.time()
            res_df.append(
                {
                    "nodes": G.number_of_nodes(),
                    "edges": G.number_of_edges(),
                    "time init": toc - tic,
                    "time optimize": toc_finished - toc,
                }
            )
            print("----------")
            print(res_df[-1])
            print("----------")
        # save updated df in every iteration
        pd.DataFrame(res_df).to_csv("outputs/runtime.csv")
