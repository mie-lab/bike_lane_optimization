import os
import time
import networkx as nx
from ebike_city_tools.random_graph import lane_to_street_graph
from ebike_city_tools.optimize.utils import output_to_dataframe, flow_to_df
from ebike_city_tools.optimize.linear_program import define_IP
from ebike_city_tools.optimize.round_simple import rounding_and_splitting


class Optimizer:
    """Generic class wrapping the optimization approaches"""

    def __init__(self, graph, od_matrix=None, shared_lane_factor=2, integer_problem=False) -> None:
        """
        graph: networkx DiGraph
        od_matrix: OD matrix as a pandas dataframe with columns (u, v, flow)
        shared_lane_factor: factor how much longer the bike travel time is on shared lanes
        """
        self.shared_lane_factor = shared_lane_factor
        self.graph = graph
        self.od_matrix = od_matrix
        self.integer_problem = integer_problem
        self.lp = None

    def init_lp(self):
        tic = time.time()
        self.lp = define_IP(
            self.graph,
            od_df=self.od_matrix,
            shared_lane_factor=self.shared_lane_factor,
            integer_problem=self.integer_problem,
        )
        print("Initialized LP", time.time() - tic)

    def optimize(self):
        assert self.lp is not None, "LP needs to initialized first, call init_lp"
        tic = time.time()
        self.lp.optimize()
        print("Finished optimizing LP", time.time() - tic)
        return self.lp.objective_value

    def postprocess(self, rounding_founction=rounding_and_splitting):
        assert self.lp is not None, "LP needs to initialized first, call init_lp"
        assert self.lp.objective_value is not None, "LP did not converged or was not optimized yet"
        # get dataframe
        capacity_values = output_to_dataframe(self.lp, self.graph)
        print("Total number of lanes possible", capacity_values["capacity"].sum() / 2)
        # apply rounding
        bike_G, car_G = rounding_founction(capacity_values.copy())
        assert nx.is_strongly_connected(car_G)
        print("Final graph edges", bike_G.number_of_edges(), car_G.number_of_edges())
        return bike_G, car_G

    def get_solution(self):
        dataframe_edge_cap = output_to_dataframe(self.lp, self.graph)
        flow_df = flow_to_df(self.lp, list(self.graph.edges))
        return dataframe_edge_cap, flow_df


def run_optimization(graph, od=None):
    """
    Simple wrapper of the whole function of Optimizer
    Run the optimization, round and return result
    """
    # need to convert the graph if it's a lane graph
    if type(graph) == nx.MultiDiGraph:
        graph = lane_to_street_graph(graph)
    optim = Optimizer(graph=graph, od_matrix=od)
    optim.init_lp()
    _ = optim.optimize()
    return optim.postprocess()
