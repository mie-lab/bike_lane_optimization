import pandas as pd
import networkx as nx
import numpy as np

from ebike_city_tools.optimize.linear_program import define_IP
from ebike_city_tools.optimize.utils import output_to_dataframe
from ebike_city_tools.optimize.round_simple import edge_to_source_target
from ebike_city_tools.utils import (
    lane_to_street_graph,
    extend_od_circular,
    compute_car_time,
    compute_edgedependent_bike_time,
)
from ebike_city_tools.iterative_algorithms import transform_car_to_bike_edge
from ebike_city_tools.metrics import od_sp

FLOW_CONSTANT = 1


class ParetoRoundOptimize:
    def __init__(self, G_lane, od, sp_method="od", optimize_every_x=5, **kwargs):
        """
        kwargs: Potential keyword arguments to be passed to the LP function
        """
        self.od = extend_od_circular(od, list(G_lane.nodes()))
        self.G_lane = G_lane
        self.sp_method = sp_method
        self.optimize_every_x = optimize_every_x
        self.optimize_kwargs = kwargs
        self.shared_lane_factor = self.optimize_kwargs.get("self.optimize_kwargs", 2)

        # transform to street graph
        self.G_street = lane_to_street_graph(G_lane)

    def optimize(self, fixed_capacities):
        """
        Returns: newly optimized capacities
        """
        ip = define_IP(self.G_street, od_df=self.od, fixed_values=fixed_capacities, **self.optimize_kwargs)
        ip.verbose = False
        ip.optimize()
        return output_to_dataframe(ip, self.G_street)

    def pareto(self):
        """
        Arguments:
            betweenness_attr: String, if car_time, we remove edges with the minimum car_time betweenness centralityk if bike_time, we
                remove edges with the highest bike_time betwenness centrality
        """
        G_lane = self.G_lane.copy()
        od_matrix = self.od
        sp_method = self.sp_method
        weight_od_flow = self.optimize_kwargs.get("weight_od_flow", False)

        # we need the car graph only to check for strongly connected
        car_graph = G_lane.copy()

        # without key but directed
        is_bike = {edge: False for edge in G_lane.edges(keys=False)}
        is_fixed_car = {edge: False for edge in G_lane.edges(keys=True)}  # TODO: get this from graph

        # set lanetype to car
        nx.set_edge_attributes(G_lane, "M", name="lanetype")

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
        # max_iters = car_graph.number_of_edges() * 10
        # iteratively add edges
        found_edge = True

        # while we still find an edge to change
        while found_edge:  # TODO not np.all(np.array(list(is_bike_or_fixed.values()))):
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
                # get one actual lane (not a street) by adding some key (does not matter which one)
                first_key = list(dict(G_lane[e[0]][e[1]]))[0]
                extended_with_key = (e[0], e[1], first_key)
                # check if this edge is already a bike lane
                if not is_fixed_car[extended_with_key]:
                    edge_to_transform = extended_with_key

                    # check if edge can be removed, if not, add it back, mark as fixed and continue
                    car_graph.remove_edge(*edge_to_transform)
                    if not nx.is_strongly_connected(car_graph):
                        car_graph.add_edge(*edge_to_transform)
                        # mark edge as car graph
                        is_fixed_car[
                            edge_to_transform
                        ] = True  # TODO: can't we also use the non-key edges for is_fixed_car?
                        # TODO: do I need to fix car capacities? to what value?
                        continue
                    else:
                        found_edge = True
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

            # compute new travel times -> TODO: move to metric computation where this is already done
            if sp_method == "od":
                bike_travel_time = od_sp(G_lane, od_matrix, weight="bike_time", weight_od_flow=weight_od_flow)
                car_travel_time = od_sp(G_lane, od_matrix, weight="car_time", weight_od_flow=weight_od_flow)
            else:
                bike_travel_time = np.mean(pd.DataFrame(nx.floyd_warshall(G_lane, weight="bike_time")).values)
                car_travel_time = np.mean(pd.DataFrame(nx.floyd_warshall(G_lane, weight="car_time")).values)
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
        return pd.DataFrame(pareto_df)


if __name__ == "__main__":
    from ebike_city_tools.random_graph import random_lane_graph, make_fake_od

    G_lane = random_lane_graph(30)
    od = make_fake_od(30, 90, nodes=G_lane.nodes)
    opt = ParetoRoundOptimize(G_lane, od)
    pareto_front = opt.pareto()
