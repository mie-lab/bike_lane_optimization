import os
import pandas as pd
import time
import networkx as nx
from ebike_city_tools.utils import lane_to_street_graph, output_to_dataframe, flow_to_df
from ebike_city_tools.optimize.linear_program import define_IP
from ebike_city_tools.optimize.round_simple import rounding_and_splitting, graph_from_integer_solution
from ebike_city_tools.optimize.iterative_rounding_and_resolving import iterative_rounding
from ebike_city_tools.optimize.randomized_rounding import randomized_rounding


class Optimizer:
    """Generic class wrapping the optimization approaches"""

    def __init__(self, graph, od_matrix=None, shared_lane_factor=2, integer_problem=False, **kwargs) -> None:
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
        self.fixed_edges = pd.DataFrame()
        self.optimizer_args = kwargs

    def init_lp(self):
        tic = time.time()
        self.lp = define_IP(
            self.graph,
            od_df=self.od_matrix,
            shared_lane_factor=self.shared_lane_factor,
            integer_problem=self.integer_problem,
            **self.optimizer_args
        )
        print("Initialized LP", time.time() - tic)

    def init_lp_with_fixed_edges(self, edge_df):
        # Auxiliary function that fixes the values for a set of edges
        tic = time.time()
        self.lp = define_IP(
            self.graph,
            fixed_edges=edge_df,
            od_df=self.od_matrix,
            shared_lane_factor=self.shared_lane_factor,
            integer_problem=self.integer_problem,
            **self.optimizer_args
        )
        print("Initialized LP with fixed edges", time.time() - tic)

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
        if self.integer_problem:
            bike_G, car_G = graph_from_integer_solution(capacity_values)
        else:
            bike_G, car_G = rounding_founction(capacity_values.copy())
        assert nx.is_strongly_connected(car_G)
        print("Final graph edges", bike_G.number_of_edges(), car_G.number_of_edges())
        return bike_G, car_G

    def get_solution(self, return_flow=False):
        dataframe_edge_cap = output_to_dataframe(self.lp, self.graph, self.fixed_edges)
        if return_flow:
            flow_df = flow_to_df(self.lp)
            return dataframe_edge_cap, flow_df
        return dataframe_edge_cap


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
