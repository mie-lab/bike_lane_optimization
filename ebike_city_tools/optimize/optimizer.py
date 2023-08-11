from typing import Any
import numpy as np
import networkx as nx
from ebike_city_tools.random_graph import city_graph, lane_to_street_graph
from ebike_city_tools.optimize.utils import output_to_dataframe
from ebike_city_tools.optimize.linear_program import initialize_IP
from ebike_city_tools.optimize.round_simple import rounding_and_splitting


class Optimizer:
    """Generic class wrapping the optimization approaches"""

    def __init__(self, round_function=rounding_and_splitting) -> None:
        self.round_function = round_function

    def __call__(self, G_city: nx.MultiDiGraph) -> Any:
        # Part 1: linear program optimization
        # convert input and optimize
        G = lane_to_street_graph(G_city)
        ip = initialize_IP(G, cap_factor=1)
        ip.optimize()

        dataframe_edge_cap = output_to_dataframe(ip, G)
        # opt_val = ip.objective_value

        # PART 2: rounding
        # test = pd.read_csv("outputs/test_lp_solution.csv")
        test = dataframe_edge_cap.copy()
        total_capacity = test["capacity"].sum() / 2
        print("Total number of lanes possible", total_capacity)

        bike_G, car_G = self.round_function(test)

        assert nx.is_strongly_connected(car_G)
        print("Final graph edges", bike_G.number_of_edges(), car_G.number_of_edges())

        return bike_G, car_G


if __name__ == "__main__":
    np.random.seed(20)

    G_city = city_graph(20)

    optim = Optimizer()
    bike_G, car_G = optim(G_city)
    assert nx.is_strongly_connected(car_G)
    print("Number of edges of output graphs: bike:", bike_G.number_of_edges(), "car:", car_G.number_of_edges())
