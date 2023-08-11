import time
import pandas as pd

from ebike_city_tools.optimize.linear_program import initialize_IP
from ebike_city_tools.optimize.utils import output_to_dataframe, make_fake_od


from ebike_city_tools.random_graph import city_graph, lane_to_street_graph

if __name__ == "__main__":
    # test on 5 different graphs for each size
    res_df = []
    for size in [30, 40, 50, 60, 70, 80, 90, 100]:
        for od_factor in [2, 3, 4]:
            for i in range(5):
                G_city = city_graph(size)
                G = lane_to_street_graph(G_city)
                od = make_fake_od(size, od_factor * size, nodes=G.nodes)

                tic = time.time()
                new_ip = initialize_IP(G, cap_factor=1, od_df=od)
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
            pd.DataFrame(res_df).to_csv("outputs/runtime.csv")
