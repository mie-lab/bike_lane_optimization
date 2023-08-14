import os
import time
import numpy as np
import pandas as pd
from ebike_city_tools.random_graph import city_graph, lane_to_street_graph
from ebike_city_tools.optimize.utils import make_fake_od
from ebike_city_tools.optimize.optimizer import Optimizer
from ebike_city_tools.optimize.utils import output_to_dataframe, flow_to_df
from ebike_city_tools.utils import add_bike_and_car_time
from ebike_city_tools.metrics import sp_length

OUT_PATH = "outputs"
os.makedirs(OUT_PATH, exist_ok=True)
NR_ITERS = 5

if __name__ == "__main__":
    np.random.seed(20)
    res_df = []
    for i in range(NR_ITERS):
        # test different number of nodes
        for size in np.arange(30, 50, 10):
            G_city = city_graph(size)
            G = lane_to_street_graph(G_city)
            # test for graphs with OD matrix of 2 times, 3 times, or 4 times as many entries as the number of nodes
            for od_factor in [3, 4, 5]:
                # define graph
                od = make_fake_od(size, od_factor * size, nodes=G.nodes)
                for integer_problem, name in zip([True, False], ["integer", "linear"]):
                    optim = Optimizer(graph=G.copy(), od_matrix=od, integer_problem=integer_problem)
                    tic = time.time()
                    optim.init_lp()
                    toc = time.time()
                    obj_value = optim.optimize()
                    toc_finished = time.time()
                    try:
                        bike_G, car_G = optim.postprocess()
                    except AssertionError:
                        print(name, "not possible because output graph is not connected")
                        continue

                    G_city = add_bike_and_car_time(G_city, bike_G, car_G)
                    bike_travel_time = sp_length(G_city, attr="biketime")
                    car_travel_time = sp_length(G_city, attr="cartime")

                    res_df.append(
                        {
                            "iter": i,
                            "name": name,
                            "nodes": size,
                            "edges": G.number_of_edges(),
                            "od_size": len(od),
                            "opt_value": obj_value,
                            "bike_time": bike_travel_time,
                            "car_time": car_travel_time,
                            "time init": toc - tic,
                            "time optimize": toc_finished - toc,
                        }
                    )

                    print("----------")
                    print(res_df[-1])
                    print("----------")
                # save updated df in every iteration
                pd.DataFrame(res_df).to_csv(os.path.join(OUT_PATH, "integer_vs_linear.csv"))
