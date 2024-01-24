import time
import os
import pandas as pd
import networkx as nx
import numpy as np

from ebike_city_tools.optimize.linear_program import define_IP
from ebike_city_tools.utils import (
    compute_car_time,
    compute_edgedependent_bike_time,
    output_to_dataframe,
    fix_multilane_bike_lanes,
)
from ebike_city_tools.graph_utils import lane_to_street_graph
from ebike_city_tools.iterative_algorithms import transform_car_to_bike_edge
from ebike_city_tools.metrics import compute_travel_times_in_graph

FLOW_CONSTANT = 1


class ParetoRoundOptimize:
    def __init__(self, G_lane, od, sp_method="od", optimize_every_x=5, **kwargs):
        """
        kwargs: Potential keyword arguments to be passed to the LP function
        """
        self.G_lane = G_lane
        assert nx.is_strongly_connected(G_lane), "Lane graph is not strongly connected"
        self.od = od
        self.sp_method = sp_method
        self.optimize_every_x = optimize_every_x
        self.optimize_kwargs = kwargs
        self.shared_lane_factor = self.optimize_kwargs.get("shared_lane_factor", 2)

        # transform to street graph
        self.G_street = lane_to_street_graph(G_lane)

        # log the runtimes for optimizing
        self.runtimes = {"time_init": [], "time_optim": []}

        # init all variables for the pareto frontier
        self.reset_pareto_variables()

    def optimize(self, fixed_capacities):
        """
        Returns: newly optimized capacities
        """
        obj_value = None
        ip = None
        counter = 0
        # increase considered number of edges until we have a valid solution
        while obj_value is None:
            if counter >= 1:
                old_valid_edges = self.optimize_kwargs.get("valid_edges_k", 0)
                assert old_valid_edges > 0, "Problem: obj value None despite no restriction on valid edges!"
                assert old_valid_edges <= 1000, "Error: stopping because no solution found even with valid edges = 1000"
                print(f"Objective value was None with valid_edges={old_valid_edges}, trying again with increased k")
                self.optimize_kwargs["valid_edges_k"] = old_valid_edges * 2
            # clean IP variable
            del ip
            # initialize
            tic = time.time()
            ip = define_IP(self.G_street, od_df=self.od, fixed_edges=fixed_capacities, **self.optimize_kwargs)
            toc = time.time()
            ip.verbose = False
            # optimize
            ip.optimize()
            obj_value = ip.objective_value
            toc_optim = time.time()
            counter += 1  # increase counter

        # log runtimes
        self.runtimes["time_init"].append(toc - tic)
        self.runtimes["time_optim"].append(toc_optim - toc)
        # create output dataframe
        optimized_capacities = output_to_dataframe(ip, self.G_street, fixed_edges=fixed_capacities)
        return optimized_capacities

    def allocate_bike_edge(self, edge_to_transform, assert_greater_0=False, remove_from_car=False):
        """
        Helper function to fix a bike edge
        -> Two use cases: for normal allocation of edges (default) or for allocating multilane bike edges-> in that case
        we need ot assert that the remaining car capacities are at least 1, and we need to remove the car edge
        """
        # Save in is_bike dictionary
        self.is_bike[edge_to_transform[:2]] = True  # lane is  bike lane
        new_edge = transform_car_to_bike_edge(self.modified_G_lane, edge_to_transform, self.shared_lane_factor)
        self.is_bike[new_edge[:2]] = True  # reversed lane is also bike lane

        # remove from car graph if not done already (only done for multiedges)
        if remove_from_car:
            self.car_graph.remove_edge(*edge_to_transform)

        # add to fixed capacities
        e = edge_to_transform[:2]
        # retrieve total capacity of this edge
        # edge_row = source_target_capacities.loc[(e[0], e[1])]
        orig_capacity = self.total_capacities[edge_to_transform[:2]]
        remaining_car_capacity = orig_capacity - 1
        # forward capacity is the lower value of the remaining capacity divided by two -> backward gets more by default
        car_capacity_straight = remaining_car_capacity // 2
        # if we can only have capacity=1 remaining, we check whether an edge even exists in the opposite direction
        # --> if no, then we put the capacity in the forward direction
        if (
            car_capacity_straight == 0
            and remaining_car_capacity > 0
            and self.car_graph.number_of_edges(edge_to_transform[1], edge_to_transform[0]) == 0
        ):
            car_capacity_straight = remaining_car_capacity
        if assert_greater_0:
            assert car_capacity_straight > 0, "remaining car capacity must be greater than 1 for multilane edges"
        self.fixed_capacities.loc[-1] = {
            "Edge": (e[1], e[0]),
            "u_b(e)": 1,
            "capacity": orig_capacity,
            "u_c(e)": remaining_car_capacity - car_capacity_straight,
        }
        self.fixed_capacities.loc[-2] = {
            "Edge": (e[0], e[1]),
            "u_b(e)": 1,
            "capacity": orig_capacity,
            "u_c(e)": car_capacity_straight,
        }
        self.fixed_capacities.index = self.fixed_capacities.index + 2

    def add_to_pareto(self, bike_edges, edges_removed):
        weight_od_flow = self.optimize_kwargs.get("weight_od_flow", False)
        # compute new travel times
        bike_travel_time, car_travel_time = compute_travel_times_in_graph(
            self.modified_G_lane, self.od, self.sp_method, weight_od_flow
        )
        assert not (pd.isna(bike_travel_time) or pd.isna(car_travel_time)), "travel times NaN"
        self.pareto_df.append(
            {
                "bike_edges_added": edges_removed,
                "bike_edges": bike_edges,
                "car_edges": self.car_graph.number_of_edges(),
                "bike_time": bike_travel_time,
                "car_time": car_travel_time,
            }
        )
        print(self.pareto_df[-1])

    def reset_pareto_variables(self):
        self.pareto_df = []

        # make copy of lane graph that we modify
        self.modified_G_lane = self.G_lane.copy()
        # set lanetype to car
        nx.set_edge_attributes(self.modified_G_lane, "M>", name="lanetype")
        # set car and bike time attributes of the graph (starting from a graph with only cars)
        car_time, bike_time = {}, {}
        for u, v, k, data in self.modified_G_lane.edges(data=True, keys=True):
            e = (u, v, k)
            car_time[e] = compute_car_time(data)
            bike_time[e] = compute_edgedependent_bike_time(data, shared_lane_factor=self.shared_lane_factor)
        nx.set_edge_attributes(self.modified_G_lane, car_time, name="car_time")
        nx.set_edge_attributes(self.modified_G_lane, bike_time, name="bike_time")

        # we need the car graph only to check for strongly connected
        self.car_graph = self.G_lane.copy()

        # whether a lane is a bike - without key but directed
        self.is_bike = {edge: False for edge in self.G_lane.edges(keys=False)}

        # initialize empty fixed capacities
        self.fixed_capacities = pd.DataFrame(columns=["Edge", "u_b(e)", "u_c(e)", "capacity"])
        self.total_capacities = nx.get_edge_attributes(self.G_street, "capacity")

    def allocate_x_bike_lanes(self, fraction_bike_lanes, fix_multilane=True):
        """Run rounding until we have allocation <fraction_bike_lanes>% of the edges as bike lanes, return graph"""
        desired_num_bike_edges = int(self.G_lane.number_of_edges() * fraction_bike_lanes)
        return self.pareto(return_graph_at_edges=desired_num_bike_edges, fix_multilane=fix_multilane)

    def pareto(
        self, save_graph_path=None, fix_multilane=True, return_list=False, return_graph_at_edges=None
    ) -> pd.DataFrame:
        """
        Computes the pareto frontier of bike and car travel times by rounding in batches
        This algorithm optimizes the capacity every x bike edges. Then, we iterate through the sorted bike capacities,
        and, if 1) the edge is not fixed yet, 2) transforming the edge into a bike lane doesn't disconnect the graph,
        the edge is allocated as a bike lane
        Arguments:
            fix_multilane: bool, determines if we initially fix one bike lane per multilane - saves computational time

        Returns:
            pareto_frontier: pd.DataFrame with columns ["bike_time", car_time", "bike_edges", "car_edges"]
        """
        self.reset_pareto_variables()

        # whether the lane is fixed as a car lane
        is_fixed_car = nx.get_edge_attributes(self.G_lane, "fixed")

        # add initial situation to pareto frontier - 0 bike edges, 0 edges added
        self.add_to_pareto(0, 0)

        # fix edges that are multilane as one bike edge
        if fix_multilane:
            edges_to_fix = fix_multilane_bike_lanes(self.G_lane, check_for_existing=False)
            # allocate them
            for e in edges_to_fix:
                self.allocate_bike_edge(e, assert_greater_0=True, remove_from_car=True)
            # add new situation to pareto frontier -> 0 actual edges added, but already x bike edges
            self.add_to_pareto(len(edges_to_fix), 0)
            print(pd.DataFrame(self.pareto_df))
        else:
            edges_to_fix = []

        edges_removed = 0
        # iteratively add edges
        found_edge = True

        # while we still find an edge to change
        while found_edge:
            # Re-optimize every x steps
            if edges_removed % self.optimize_every_x == 0:
                # Run optimization
                capacities = self.optimize(self.fixed_capacities)
                # sort capacities by bike and car capacities -> first consider the ones with high bike and low car capacity
                cap_sorted = capacities.sort_values(["u_b(e)", "u_c(e)"], ascending=[False, True])

            found_edge = False
            # iterate over capacities until we find an edge that can be added
            for e in cap_sorted["Edge"]:
                # check if e is even in the original graph -> otherwise wait for iteration with reversed edge
                # also check if e is a bike lane already
                if (e not in self.modified_G_lane.edges()) or self.is_bike[e]:
                    continue
                # iterate over the lanes of this edge and try to find one that can be converted
                for key in list(dict(self.modified_G_lane[e[0]][e[1]])):
                    edge_to_transform = (e[0], e[1], key)
                    # check if this edge is already fixed
                    if not is_fixed_car.get(edge_to_transform, False):
                        # check if edge can be removed, if not, add it back, mark as fixed and continue
                        self.car_graph.remove_edge(*edge_to_transform)
                        if not nx.is_strongly_connected(self.car_graph):
                            self.car_graph.add_edge(*edge_to_transform)
                            # mark edge as car graph
                            is_fixed_car[edge_to_transform] = True
                            continue
                        else:
                            found_edge = True
                            break
                # stop when we found an edge that can be converted to a bike lane
                if found_edge:
                    break
            # make sure that we stop and don't remove the last edge
            if not found_edge:
                break

            # if it can be removed, we transform the travel times
            edges_removed += 1
            # transform to bike lane -> update bike and car time
            self.allocate_bike_edge(edge_to_transform)
            # update pareto frontier
            self.add_to_pareto(len(edges_to_fix) + edges_removed, edges_removed)

            # save graph with the same frequency as re-optimizing (always saved before reoptimizing)
            if edges_removed % self.optimize_every_x == 0:
                # always save the intermediate graph from a specific point onwards
                if save_graph_path is not None and edges_removed > 20:
                    # convert to dataframe
                    edge_df = nx.to_pandas_edgelist(self.modified_G_lane, edge_key="edge_key")[
                        ["source", "target", "edge_key", "fixed", "lanetype", "distance", "gradient", "speed_limit"]
                    ]
                    edge_df.to_csv(save_graph_path + f"_graph_{edges_removed}.csv", index=False)

            # return graph if at this number of edges
            if return_graph_at_edges is not None and edges_removed == return_graph_at_edges:
                return self.modified_G_lane

        # if we have reached the end but not the number of edges we wanted to allocate, return graph
        if return_graph_at_edges is not None:
            return self.modified_G_lane
        if return_list:
            return self.pareto_df
        return pd.DataFrame(self.pareto_df)
