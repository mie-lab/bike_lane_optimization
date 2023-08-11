import os
import time

from ebike_city_tools.optimize.optimizer import Optimizer
from ebike_city_tools.optimize.utils import output_to_dataframe, flow_to_df

OUT_PATH = "outputs"
os.makedirs(OUT_PATH, exist_ok=True)

if __name__ == "__main__":
    from ebike_city_tools.random_graph import city_graph, lane_to_street_graph
    from ebike_city_tools.optimize.utils import make_fake_od
    import numpy as np

    np.random.seed(20)

    G_city = city_graph(20)
    G = lane_to_street_graph(G_city)

    od = make_fake_od(20, 40, nodes=G.nodes)
    # # for flow-sum-constraint: compute maximal number of paths that could pass through one edge
    # max_paths_one_edge = len(od)  # maximum number of paths through one edge corresponds to number of OD-pairs
    # FACTOR_MAX_PATHS = 0.5  # only half of the paths are allowed to use the same street
    # cap_factor = max_paths_one_edge * FACTOR_MAX_PATHS

    optim = Optimizer(graph=G, od_matrix=od)
    optim.init_lp()
    obj_value = optim.optimize()
    dataframe_edge_cap, flow_df = optim.get_solution()

    flow_df.to_csv(os.path.join(OUT_PATH, "test_flow_solution.csv"), index=False)
    dataframe_edge_cap.to_csv(os.path.join(OUT_PATH, "test_lp_solution.csv"), index=False)
    print("OPT VALUE", obj_value)
    # save the new graph --> undirected
    # nx.write_gpickle(G, "outputs/test_G_random.gpickle")
