import time
import pandas as pd
import networkx as nx
import numpy as np

from ebike_city_tools.optimize.linear_program import define_IP
from ebike_city_tools.optimize.round_simple import edge_to_source_target
from ebike_city_tools.utils import (
    lane_to_street_graph,
    extend_od_circular,
    compute_car_time,
    compute_edgedependent_bike_time,
    output_to_dataframe,
)
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

    def optimize(self, fixed_capacities):
        """
        Returns: newly optimized capacities
        """
        tic = time.time()
        ip = define_IP(self.G_street, od_df=self.od, fixed_edges=fixed_capacities, **self.optimize_kwargs)
        toc = time.time()
        self.runtimes["time_init"].append(toc - tic)
        ip.verbose = False
        ip.optimize()
        toc_optim = time.time()
        self.runtimes["time_optim"].append(toc_optim - toc)
        return output_to_dataframe(ip, self.G_street, fixed_edges=fixed_capacities)

    def pareto(self, max_bike_edges=np.inf, return_list=False, return_graph=False) -> pd.DataFrame:
        """
        Computes the pareto frontier of bike and car travel times by rounding in batches
        This algorithm optimizes the capacity every x bike edges. Then, we iterate through the sorted bike capacities,
        and, if 1) the edge is not fixed yet, 2) transforming the edge into a bike lane doesn't disconnect the graph,
        the edge is allocated as a bike lane
        Returns:
            pareto_frontier: pd.DataFrame with columns ["bike_time", car_time", "bike_edges", "car_edges"]
        """
        G_lane = self.G_lane.copy()
        weight_od_flow = self.optimize_kwargs.get("weight_od_flow", False)

        # we need the car graph only to check for strongly connected
        car_graph = G_lane.copy()

        # without key but directed
        is_bike = {edge: False for edge in G_lane.edges(keys=False)}
        is_fixed_car = nx.get_edge_attributes(G_lane, "fixed")

        # set lanetype to car
        nx.set_edge_attributes(G_lane, "M>", name="lanetype")

        # set car and bike time attributes of the graph (starting from a graph with only cars)
        car_time, bike_time = {}, {}
        for u, v, k, data in G_lane.edges(data=True, keys=True):
            e = (u, v, k)
            car_time[e] = compute_car_time(data)
            bike_time[e] = compute_edgedependent_bike_time(data, shared_lane_factor=self.shared_lane_factor)
        nx.set_edge_attributes(G_lane, car_time, name="car_time")
        nx.set_edge_attributes(G_lane, bike_time, name="bike_time")

        # optimize without fixed capacities
        fixed_capacities = pd.DataFrame(columns=["Edge", "u_b(e)", "u_c(e)", "capacity"])

        # initialize pareto result list
        pareto_df = []

        edges_removed = 0
        # iteratively add edges
        found_edge = True

        # while we still find an edge to change
        while found_edge and edges_removed < max_bike_edges:
            if edges_removed % self.optimize_every_x == 0:
                # Run optimization
                capacities = self.optimize(fixed_capacities)
                # sort capacities by bike and car capacities -> first consider the ones with high bike and low car capacity
                cap_sorted = capacities.sort_values(["u_b(e)", "u_c(e)"], ascending=[False, True])

            found_edge = False
            # iterate over capacities until we find an edge that can be added
            for e in cap_sorted["Edge"]:
                # check if e is even in the original graph -> otherwise wait for iteration with reversed edge
                # also check if e is a bike lane already
                if (e not in G_lane.edges()) or is_bike[e]:
                    continue
                # iterate over the lanes of this edge and try to find one that can be converted
                for key in list(dict(G_lane[e[0]][e[1]])):
                    edge_to_transform = (e[0], e[1], key)
                    # check if this edge is already fixed
                    if not is_fixed_car.get(edge_to_transform, False):
                        # check if edge can be removed, if not, add it back, mark as fixed and continue
                        car_graph.remove_edge(*edge_to_transform)
                        if not nx.is_strongly_connected(car_graph):
                            car_graph.add_edge(*edge_to_transform)
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
            # transform to bike lane -> update bike and car time
            edges_removed += 1
            is_bike[edge_to_transform[:2]] = True  # lane is  bike lane
            new_edge = transform_car_to_bike_edge(G_lane, edge_to_transform, self.shared_lane_factor)
            is_bike[new_edge[:2]] = True  # reversed lane is also bike lane

            # add to fixed capacities
            e = edge_to_transform[:2]
            source_target_capacities = edge_to_source_target(capacities).set_index(["source", "target"])
            edge_row = source_target_capacities.loc[(e[0], e[1])]
            edge_reversed_row = source_target_capacities.loc[(e[0], e[1])]
            orig_capacity = edge_row["capacity"]
            car_capacity = edge_row["capacity"] - 1
            car_capacity_straight = car_capacity // 2  # divide between straight and reversed -> reversed gets more
            fixed_capacities.loc[-1] = {
                "Edge": (e[1], e[0]),
                "u_b(e)": 1,
                "capacity": orig_capacity,
                "u_c(e)": car_capacity - car_capacity_straight,
            }
            fixed_capacities.loc[-2] = {
                "Edge": (e[0], e[1]),
                "u_b(e)": 1,
                "capacity": orig_capacity,
                "u_c(e)": car_capacity_straight,
            }
            fixed_capacities.index = fixed_capacities.index + 2

            # compute new travel times
            bike_travel_time, car_travel_time = compute_travel_times_in_graph(
                G_lane, self.od, self.sp_method, weight_od_flow
            )
            assert not (pd.isna(bike_travel_time) or pd.isna(car_travel_time)), "travel times NaN"
            pareto_df.append(
                {
                    "bike_edges_added": edges_removed,
                    "bike_edges": edges_removed,
                    "car_edges": car_graph.number_of_edges(),
                    "bike_time": bike_travel_time,
                    "car_time": car_travel_time,
                }
            )
            print(pareto_df[-1])

        if return_graph:
            return G_lane
        if return_list:
            return pareto_df
        return pd.DataFrame(pareto_df)
