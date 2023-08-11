import time
import os
import pandas as pd
from ebike_city_tools.iterative_algorithms import *
from ebike_city_tools.random_graph import *
from ebike_city_tools.visualize import *
from ebike_city_tools.metrics import *
from ebike_city_tools.optimize.optimizer import run_optimization
from ebike_city_tools.utils import add_bike_and_car_time

ITERS_TEST = 3
OUT_PATH = "outputs"
os.makedirs(OUT_PATH, exist_ok=True)

# All algorithms that we test
algorithm_dict = {
    # "spanning_random": {"bike": extract_spanning_tree, "car": random_edge_order},
    # "full_random": {"bike": extract_oneway_subnet, "car": random_edge_order},
    # "spanning_balanced": {"bike": extract_spanning_tree, "car": greedy_nodes_balanced},
    # "full_balanced": {"bike": extract_oneway_subnet, "car": greedy_nodes_balanced},
    "betweenness": {"bike_and_car": greedy_betweenness},
    # "optim_betweenness": {"bike_and_car": optimized_betweenness},
    "optimize": {"bike_and_car": run_optimization},
}
# metrics to evaluate
metrics_for_eval = ["sp_reachability", "sp_hops", "closeness"]
bike_car_metrics = ["bike_" + m for m in metrics_for_eval] + ["car_" + m for m in metrics_for_eval]

all_results = []
for i in range(ITERS_TEST):
    # collect results in a dataframe
    df = pd.DataFrame(index=list(algorithm_dict.keys()) + ["original"], columns=bike_car_metrics + ["runtime"])

    # generate a random (street-network realistic) graph
    base_graph = city_graph()  # generate_base_graph()

    # Run all algorithms on this base graph and generate bike and car graph
    for algorithm in algorithm_dict.keys():
        # either the algorithm does bike and car lane graph extraction at the same time
        tic = time.time()
        if "bike_and_car" in algorithm_dict[algorithm]:
            # this option is for algorithms that optimize bike- and car-lane graph at the same time
            bike_and_car_algo = algorithm_dict[algorithm]["bike_and_car"]  # get algorithm function from dict
            bike_graph, car_graph = bike_and_car_algo(base_graph)  # apply algorithm to the data
        else:
            # or first the bike lane graph is optimized, and then the car lane graph is adapted as good as possible
            bike_algo = algorithm_dict[algorithm]["bike"]
            # split original network into bike lanes and the leftover part (undirected)
            bike_graph, leftover_car_graph = bike_algo(base_graph)
            # get algorithm for car network processing
            car_algo = algorithm_dict[algorithm]["car"]
            # improve car network
            car_graph = car_algo(leftover_car_graph)
        runtime = time.time() - tic

        # Compute all metrics for both bike- and car lane graph
        for graph_name, graph_to_eval in zip(["bike", "car"], [bike_graph, car_graph]):
            for metric in metrics_for_eval:
                df.loc[algorithm, graph_name + "_" + metric] = eval(metric)(graph_to_eval)
            df.loc[algorithm, "runtime"] = runtime

        # Compute weightest shortest path length for both
        G_city = add_bike_and_car_time(base_graph, bike_graph, car_graph)
        df.loc[algorithm, "bike_travel_time"] = sp_length(G_city, attr="biketime")
        df.loc[algorithm, "car_travel_time"] = sp_length(G_city, attr="cartime")

        # add metrics for original graph (pretending that we did not subtract any car lanes)
        for metric in metrics_for_eval:
            df.loc["original", "car_" + metric] = eval(metric)(base_graph)
            df.loc["original", "runtime"] = 0
            G_city = add_bike_and_car_time(base_graph, nx.Graph(), base_graph)
            df.loc["original", "car_travel_time"] = sp_length(G_city, attr="cartime")
            # df.loc["original", "bike_" + metric] = 0 # could fill with 0 for some metrics and inf for others

    df.index.name = "Method"
    df.reset_index(inplace=True)
    df["run"] = i
    all_results.append(df)

# Average over all runs and sort by closeness metric
df = pd.concat(all_results)
# df = df.drop("run", axis=1).groupby("Method").mean()

# Save results as a csv
df.to_csv(os.path.join(OUT_PATH, "algo_comp.csv"))

# Plotting
# scatter_car_bike(df, metrics_for_eval, OUT_PATH)