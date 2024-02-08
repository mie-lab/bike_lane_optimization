import networkx as nx
import pandas as pd
import numpy as np
from collections import defaultdict

from ebike_city_tools.utils import (
    compute_edgedependent_bike_time,
    compute_car_time,
    compute_penalized_car_time,
    fix_multilane_bike_lanes,
)
from ebike_city_tools.graph_utils import lossless_to_undirected


def extract_spanning_tree(G):
    """
    Find the minimum spanning tree in G (according to edge weights) and subtract it
    Returns:
        spanning_tree: The undirected minimum spanning tree
        left_over_G: an undirected version of the remaining (multi-) graph
    """
    undirected_G = lossless_to_undirected(G)
    assert G.number_of_edges() == undirected_G.number_of_edges(), "lost edges"
    spanning_tree = nx.minimum_spanning_tree(undirected_G)

    # subtract T from undirected multigraph
    left_over_G = undirected_G.copy()
    left_over_G.remove_edges_from(spanning_tree.edges())

    #     print("Spanning tree edges", spanning_tree.number_of_edges(), "leftover edges", left_over_G.number_of_edges(), "original edges", G.number_of_edges())
    return spanning_tree, left_over_G


def extract_oneway_subnet(G):
    """
    Extract the bike lane graph as a full undirected version of the whole network, by subtracting in from the multigraph
    Returns:
        bike_lane_graph: The undirected graph covering all streets
        left_over_G: an undirected version of the remaining (multi-) graph
    """
    undirected_G = lossless_to_undirected(G)
    assert G.number_of_edges() == undirected_G.number_of_edges(), "lost edges"

    bike_lane_graph = nx.Graph(undirected_G)

    left_over_G = undirected_G.copy()
    left_over_G.remove_edges_from(bike_lane_graph.edges())

    #     print("Spanning tree edges", bike_lane_graph.number_of_edges(), "leftover edges", left_over_G.number_of_edges(), "original edges", G.number_of_edges())
    return bike_lane_graph, left_over_G


def random_edge_order(undirected_car_G):
    """Random baseline for optimizing the car graph: Simply assign each edge a random direction"""
    new_edge_list = []
    for e in undirected_car_G.edges():
        if np.random.rand() < 0.5:
            new_edge_list.append((e[0], e[1], {"weight": 1}))
        else:
            new_edge_list.append((e[1], e[0], {"weight": 1}))

    # temp variable for orig node attributes
    node_attributes = nx.get_node_attributes(undirected_car_G, name="loc")

    # directed edges
    car_G = nx.MultiDiGraph()
    car_G.add_edges_from(new_edge_list)

    # set attributes
    nx.set_node_attributes(car_G, node_attributes, name="loc")

    return car_G


def greedy_nodes_balanced(undirected_car_G):
    """
    Bad baseline for optimizing the car graph:
    Iteratively assign edge direction such that 1) the in- and outgoing edges are roughly balanced, and 2) all remaining
    multi-edges are reciprocal
    """
    current_out = defaultdict(int)
    current_in = defaultdict(int)

    # save node attributes for new graph later
    node_attributes = nx.get_node_attributes(undirected_car_G, name="loc")

    directed_edge_list = []
    car_G_edges = list(undirected_car_G.edges())

    queue = [(i, e) for i, e in enumerate(car_G_edges) if e[0] == 0]
    done_edges = set([q[0] for q in queue])

    # Set edge directions
    while len(queue) > 0:
        edge_ind, edge = queue.pop(0)
        #     print(edge_ind, edge)
        if current_out[edge[0]] > current_in[edge[0]]:
            directed_edge_list.append([edge[1], edge[0], {"weight": 1}])
            current_in[edge[0]] += 1
            current_out[edge[1]] += 1
        else:
            directed_edge_list.append([edge[0], edge[1], {"weight": 1}])
            current_in[edge[1]] += 1
            current_out[edge[0]] += 1
        for i, e in enumerate(car_G_edges):
            if i not in done_edges and edge[1] in e:
                queue.append((i, e))
                done_edges.add(i)

        if len(queue) == 0 and (len(directed_edge_list) < len(car_G_edges)):
            leftover = [k for k in np.arange(len(car_G_edges)) if k not in done_edges][0]
            queue.append((leftover, car_G_edges[leftover]))
            done_edges.add(leftover)

    assert len(directed_edge_list) == len(car_G_edges), f"{len(directed_edge_list)}, {len(car_G_edges)}"

    car_G = nx.MultiDiGraph()
    car_G.add_edges_from(directed_edge_list)

    # set attributes
    nx.set_node_attributes(car_G, node_attributes, name="loc")

    return car_G


def greedy_betweenness(lane_graph_inp, bike_edges_to_add=None):
    """
    Algorithm by Lukas based on Steinacker et al (2022)
    Iteratively remove the edges with lowest betweenness centrality, and add them to the bike lane network if they do
    not destroy strong connectivity
    """
    # (such that only one lane) per street can be selected
    # copy graph
    lane_graph = lane_graph_inp.copy()
    # save nodes for bike graph later
    node_attributes = nx.get_node_attributes(lane_graph, name="loc")
    # init is_fixed
    is_fixed = {edge: False for edge in lane_graph.edges}

    iters, edges_removed = 0, 0
    # max iters
    max_iters = lane_graph.number_of_edges() * 10
    bike_edges = []
    if bike_edges_to_add is None:
        # if None, remove half of the edges or as many as possible
        bike_edges_to_add = int(0.5 * lane_graph.number_of_edges())

    # iteratively recompute betweenness centrality
    while iters < max_iters:
        betweenness = nx.edge_betweenness_centrality(lane_graph)
        # find edge with lowest betweenness centrality that it not fixed
        sorted_edges = sorted(betweenness.items(), key=lambda x: x[1])
        for s in sorted_edges:
            if not is_fixed[s[0]]:
                min_edge = s[0]
                break
        # remove this edge
        lane_graph.remove_edge(*min_edge)
        if not nx.is_strongly_connected(lane_graph):
            lane_graph.add_edge(*min_edge)
            is_fixed[min_edge] = True
            if all(is_fixed.values()):
                break
        else:
            edges_removed += 1
            bike_edges.append(min_edge)
        iters += 1
        if edges_removed >= bike_edges_to_add:
            # print("half of the edges are removed, break")
            break

    bike_graph = nx.MultiGraph()
    multi_bike_edge_list = [[e[0], e[1], i, {}] for i, e in enumerate(bike_edges)]
    bike_graph.add_edges_from(multi_bike_edge_list)
    nx.set_node_attributes(bike_graph, node_attributes, name="loc")

    # print("is fixed", sum(is_fixed.values()))
    assert nx.is_strongly_connected(lane_graph)
    # print(len(lane_graph.edges()), len(lane_graph_inp.edges()), nx.is_strongly_connected(lane_graph))

    return bike_graph, lane_graph


def od_betweenness_and_splength(G_lane, od_matrix, attr, weight_od_flow=False):
    """
    Own implementation of computing betweenness centrality
    Computes the shortest path between each OD pair (weighted by bike_time or car_time (attr))

    """
    # weight
    dist_per_edge = nx.get_edge_attributes(G_lane, attr)
    # set centrality to 0 in the beginnig
    edge_centrality = {e: 0 for e in G_lane.edges(keys=True)}
    # iterate over OD matrix
    sp_lengths = []
    for s, t, st_weight in od_matrix.values:
        # compute SP
        shortest_path = nx.shortest_path(G_lane, source=s, target=t, weight=attr)
        path_length = 0
        # iterate overb path and aggregate betwenness and path length
        for i in range(len(shortest_path) - 1):
            # for each edge on the path, get the edge (from multiedges) that has the lowest attr value
            # Case 1: car time: we remove the edge with lowest car betweenness centrality. Therefore, we give all the
            # betweenness centrality to the one with lower car time. Once one edge is turned into a bike lane, the other
            # one should survive.
            # Case 2: bike time: We remove the edge with the highest bike betweenness centrality. We give all the
            # betweenness centrality to the one with lower bike time, so that this edge is prioritized. Once one edge is
            # turned into a bike lane, it will keep on having high centrality to prevent the other edge from dropping
            # Case 3: top down: We remove the edge with the lowest bike betweenness centrality.
            min_val = np.inf
            for key, key_dict in G_lane[shortest_path[i]][shortest_path[i + 1]].items():  # this gives the keys
                if key_dict[attr] < min_val:
                    min_key = key
                    min_val = key_dict[attr]
            if min_val == np.inf:
                # min_key = key  # uncomment to allow infinite SP
                raise RuntimeError("inf should not appear on SP")
            # print(shortest_path[i], shortest_path[i + 1], first_key)
            edge = (shortest_path[i], shortest_path[i + 1], min_key)
            if weight_od_flow:
                edge_centrality[edge] += st_weight
                path_length += dist_per_edge[edge] * st_weight
            else:
                edge_centrality[edge] += 1
                path_length += dist_per_edge[edge]
        sp_lengths.append(path_length)
    return edge_centrality, np.mean(sp_lengths)


def compute_betweenness_and_splength(
    G_lane, betweenness_attr, od_matrix=None, sp_method="all_pairs", weight_od_flow=False
):
    """
    Compute betweenness centrality and average shortest path length either for all pairs or OD pairs
    """
    if sp_method == "all_pairs":
        # could be combined but not sure if it's worth it in my case
        car_travel_time = np.mean(pd.DataFrame(nx.floyd_warshall(G_lane, weight="car_time")).values)
        bike_travel_time = np.mean(pd.DataFrame(nx.floyd_warshall(G_lane, weight="bike_time")).values)
        betweenness = nx.edge_betweenness_centrality(G_lane, weight=betweenness_attr)
    else:
        # manually compute travel times and edge betweenness centrality on the OD paths
        betweenness_car, car_travel_time = od_betweenness_and_splength(
            G_lane, od_matrix, "car_time", weight_od_flow=weight_od_flow
        )
        # assert travel_times_car == od_sp(  # debugging
        #     G_lane, od_matrix, "car_time", weight_od_flow=False
        # ), "mismatch between new and old computation"
        betweenness_bike, bike_travel_time = od_betweenness_and_splength(
            G_lane, od_matrix, "bike_time", weight_od_flow=weight_od_flow
        )
        betweenness = betweenness_car if betweenness_attr == "car_time" else betweenness_bike
    return betweenness, car_travel_time, bike_travel_time


def betweenness_pareto(
    G_lane,
    od_matrix=None,
    sp_method="all_pairs",
    shared_lane_factor=2,
    weight_od_flow=False,
    fix_multilane=False,
    betweenness_attr="car_time",
    save_graph_path=None,
    save_graph_every_x=50,
    return_graph_at_edges=None,
):
    """
    Arguments:
        betweenness_attr: String, if car_time, we remove edges with the minimum car_time betweenness centralityk if bike_time, we
            remove edges with the highest bike_time betwenness centrality
    """
    # initialize pareto
    pareto_df = []
    assert betweenness_attr in ["car_time", "bike_time"]
    # we need the car graph only to check for strongly connected
    car_graph = G_lane.copy()
    # get fixed attribute
    is_bike_or_fixed = nx.get_edge_attributes(G_lane, "fixed")

    # set lanetype to car for all edges initially
    nx.set_edge_attributes(G_lane, "M>", name="lanetype")

    def add_to_pareto(bike_edges, added_edges):
        # compute SP wrt car time or bike time and wrt OD matrix possibly
        betweenness, car_travel_time, bike_travel_time = compute_betweenness_and_splength(
            G_lane, betweenness_attr, od_matrix=od_matrix, sp_method=sp_method, weight_od_flow=weight_od_flow
        )
        pareto_df.append(
            {
                "bike_edges_added": added_edges,
                "bike_edges": bike_edges,
                "car_edges": car_graph.number_of_edges(),
                "bike_time": bike_travel_time,
                "car_time": car_travel_time,
            }
        )
        return betweenness

    # set car and bike time attributes of the graph (starting from a graph with only cars)
    car_time, bike_time = {}, {}
    for u, v, k, data in G_lane.edges(data=True, keys=True):
        e = (u, v, k)
        car_time[e] = compute_car_time(data)
        bike_time[e] = compute_edgedependent_bike_time(data, shared_lane_factor=shared_lane_factor)
    nx.set_edge_attributes(G_lane, car_time, name="car_time")
    nx.set_edge_attributes(G_lane, bike_time, name="bike_time")

    # add first entry to pareto frontier with 0 edges added
    betweenness = add_to_pareto(0, 0)
    print(pareto_df[-1])

    # fix edges that are multilane as one bike edge
    if fix_multilane:
        edges_to_fix = fix_multilane_bike_lanes(G_lane, check_for_existing=False)
        # allocate them
        for edge_to_transform in edges_to_fix:
            if is_bike_or_fixed[edge_to_transform]:
                continue
            # check if edge can be removed, if not, add it back, mark as fixed and continue
            car_graph.remove_edge(*edge_to_transform)
            assert nx.is_strongly_connected(car_graph)
            # transform to bike lane -> update bike and car time
            new_edge = transform_car_to_bike_edge(G_lane, edge_to_transform, shared_lane_factor)
            # mark edge as checked
            is_bike_or_fixed[edge_to_transform] = True
            is_bike_or_fixed[new_edge] = True
            # # fix the other lane as a car edge
            # multiedge_dict = dict(G_lane[edge_to_transform[0]][edge_to_transform[1]])
            # del multiedge_dict[edge_to_transform[2]]  # remove key
            # second_key = list(multiedge_dict.keys())[0]
            # is_bike_or_fixed[(edge_to_transform[0], edge_to_transform[1], second_key)] = True
        # add new situation to pareto frontier -> 0 actual edges added, but already x bike edges
        betweenness = add_to_pareto(len(edges_to_fix), 0)
        print(pd.DataFrame(pareto_df))
    else:
        edges_to_fix = []

    edges_removed = 0
    # max_iters = car_graph.number_of_edges() * 10
    # iteratively add edges until no edge is found anymore
    edge_found = True
    while edge_found:
        # sort betweenens centrality (use highest centrality if bike_time, so reverse)
        sorted_by_betweenness = sorted(betweenness.items(), key=lambda x: x[1], reverse=betweenness_attr == "bike_time")
        edge_found = False
        for s in sorted_by_betweenness:
            if not is_bike_or_fixed.get(s[0], False):
                edge_to_transform = s[0]
                edge_found = True
                break
        if not edge_found:
            break

        # mark edge as checked
        is_bike_or_fixed[edge_to_transform] = True
        # check if edge can be removed, if not, add it back, mark as fixed and continue
        car_graph.remove_edge(*edge_to_transform)
        if not nx.is_strongly_connected(car_graph):
            car_graph.add_edge(*edge_to_transform)
            continue

        # if it can be removed, we transform the travel times
        # transform to bike lane -> update bike and car time
        edges_removed += 1
        new_edge = transform_car_to_bike_edge(G_lane, edge_to_transform, shared_lane_factor)
        is_bike_or_fixed[new_edge] = True

        # add to pareto frontier
        betweenness = add_to_pareto(len(edges_to_fix) + edges_removed, edges_removed)

        if return_graph_at_edges == edges_removed:
            return G_lane

        # save graph
        if save_graph_path is not None and edges_removed % save_graph_every_x == 0:
            edge_df = nx.to_pandas_edgelist(G_lane, edge_key="edge_key")[
                ["source", "target", "edge_key", "fixed", "lanetype", "distance", "gradient", "speed_limit"]
            ]
            edge_df.to_csv(save_graph_path + f"_graph_{edges_removed}.csv", index=False)

        print(pareto_df[-1])

    # if we have not achieved the desired edge count, we still return the graph at its current state
    if return_graph_at_edges is not None:
        return G_lane
    return pd.DataFrame(pareto_df)


def transform_car_to_bike_edge(G_lane, edge_to_transform, shared_lane_factor):
    # transform the edge into a bike lane
    G_lane.edges[edge_to_transform]["lanetype"] = "P"
    G_lane.edges[edge_to_transform]["car_time"] = np.inf
    # # debugging:
    # new_bike_time = compute_edgedependent_bike_time(
    #     G_lane.edges[edge_to_transform], shared_lane_factor=shared_lane_factor
    # )
    # assert new_bike_time == G_lane.edges[edge_to_transform]["bike_time"] / shared_lane_factor
    G_lane.edges[edge_to_transform]["bike_time"] = G_lane.edges[edge_to_transform]["bike_time"] / shared_lane_factor

    # insert helper edge in the opposite direction - same distance etc
    new_edge_attrs = G_lane.edges[edge_to_transform].copy()
    # change gradient direction
    new_edge_attrs["gradient"] = -new_edge_attrs["gradient"]
    # compute new bike time based on new gradient
    new_edge_attrs["bike_time"] = compute_edgedependent_bike_time(new_edge_attrs, shared_lane_factor=shared_lane_factor)
    # se tnew edge_key
    new_edge = (edge_to_transform[1], edge_to_transform[0], f"{edge_to_transform[1]}-{edge_to_transform[0]}-revbike")
    G_lane.add_edge(*new_edge, **new_edge_attrs)
    return new_edge


def optimized_betweenness(lane_graph_inp, nr_iters=1000):
    from ebike_city_tools.rl_env import StreetNetworkEnv

    _, car_graph = greedy_betweenness(lane_graph_inp)
    env = StreetNetworkEnv(lane_graph_inp)
    env.derive_initial_setting(car_graph)
    env.reset()
    # print("start_reward", env.last_reward)
    prev_reward = env.last_reward
    for _ in range(nr_iters):
        act_avail = env.get_available_actions()
        #     action = np.random.randint(env.n_actions)
        action = np.random.choice(act_avail)
        rew = env.step(action)
        # revert action if it didn't help
        if rew < prev_reward:
            rew = env.revert_action(action)
            # assert rew >= prev_reward
        prev_reward = rew
    # print("final reward", rew)
    return env.bike_graph, env.car_graph


def topdown_betweenness_pareto(
    G_lane,
    od_matrix=None,
    sp_method="all_pairs",
    shared_lane_factor=2,
    fix_multilane=False,
    weight_od_flow=False,
    save_graph_path=None,
    save_graph_every_x=50,
):
    """
    Implements the algorithm from Steinacker et al where we start with a full bike network and iteratively remove bike
    lanes
    In constrast to the original implementation, we still assume an impact on the car network -> assume 10kmh speed on
    bike priority lanes
    """

    # all edges are bike edges
    is_car_edge = {e: False for e in G_lane.edges(keys=True)}
    nr_edges = G_lane.number_of_edges()

    # set lanetype to bike -> all car times are inf
    nx.set_edge_attributes(G_lane, "P", name="lanetype")

    # set bike time attributes of the graph (starting from a graph with only cars)
    bike_time, car_time = {}, {}
    for u, v, k, data in G_lane.edges(data=True, keys=True):
        e = (u, v, k)
        bike_time[e] = compute_edgedependent_bike_time(data, shared_lane_factor=shared_lane_factor)
        car_time[e] = compute_penalized_car_time(data)
    nx.set_edge_attributes(G_lane, bike_time, name="bike_time")
    nx.set_edge_attributes(G_lane, car_time, name="car_time")

    # if multilane: get the set of edges that should never be transformed into car edges
    if fix_multilane:
        edges_to_fix = fix_multilane_bike_lanes(G_lane, check_for_existing=False)
        for e in edges_to_fix:
            # just pretend that they are car edges without transforming them --> will never be transformed
            is_car_edge[e] = True

    # compute betweenness and bike travel time
    if sp_method == "all_pairs":
        betweenness = nx.edge_betweenness_centrality(G_lane, weight="bike_time")
    elif sp_method == "od":
        betweenness, bike_travel_time = od_betweenness_and_splength(
            G_lane, od_matrix, "bike_time", weight_od_flow=weight_od_flow
        )
    else:
        raise NotImplementedError("only all_pairs or od allowed for sp_metho")

    pareto_df = []
    edges_removed = 0
    while not np.all(np.array(list(is_car_edge.values()))):
        # sort betweenens centrality (use highest centrality if bike_time, so reverse)
        sorted_by_betweenness = sorted(betweenness.items(), key=lambda x: x[1])
        for s in sorted_by_betweenness:
            if not is_car_edge[s[0]]:
                edge_to_transform = s[0]
                break
        edges_removed += 1
        G_lane.edges[edge_to_transform]["lanetype"] = "M>"
        G_lane.edges[edge_to_transform]["car_time"] = compute_car_time(G_lane.edges[edge_to_transform])
        # increase bike_time
        G_lane.edges[edge_to_transform]["bike_time"] *= shared_lane_factor
        is_car_edge[edge_to_transform] = True

        # compute new travel times
        betweenness, car_travel_time, bike_travel_time = compute_betweenness_and_splength(
            G_lane, "bike_time", od_matrix=od_matrix, sp_method=sp_method, weight_od_flow=weight_od_flow
        )
        pareto_df.append(
            {
                "bike_edges_added": nr_edges - edges_removed,
                "bike_edges": nr_edges - edges_removed,
                "car_edges": edges_removed,
                "bike_time": bike_travel_time,
                "car_time": car_travel_time,
            }
        )
        print(pareto_df[-1])

        # save graph
        if save_graph_path is not None and (edges_removed % save_graph_every_x == 0):
            edge_df = nx.to_pandas_edgelist(G_lane, edge_key="edge_key")[
                ["source", "target", "edge_key", "fixed", "lanetype", "distance", "gradient", "speed_limit"]
            ]
            edge_df.to_csv(save_graph_path + f"_graph_{edges_removed}.csv", index=False)
    return pd.DataFrame(pareto_df)
