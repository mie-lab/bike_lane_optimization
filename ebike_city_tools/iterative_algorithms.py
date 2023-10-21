import networkx as nx
import pandas as pd
import numpy as np
from collections import defaultdict

from ebike_city_tools.utils import lossless_to_undirected, compute_edgedependent_bike_time, compute_car_time


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
            # TODO: divide centrality between several keys? makes sense for cars, not so much for bikes
            min_val = np.inf
            for key, key_dict in G_lane[shortest_path[i]][shortest_path[i + 1]].items():  # this gives the keys
                if key_dict[attr] < min_val:
                    min_key = key
                    min_val = key_dict[attr]
            if min_val == np.inf:
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
    betweenness_attr="car_time",
):
    """
    Arguments:
        betweenness_attr: String, if car_time, we remove edges with the minimum car_time betweenness centralityk if bike_time, we
            remove edges with the highest bike_time betwenness centrality
    """
    assert betweenness_attr in ["car_time", "bike_time"]
    # we need the car graph only to check for strongly connected
    car_graph = G_lane.copy()

    is_bike_or_fixed = {edge: False for edge in G_lane.edges}  # TODO: get edge attribute "fixed"

    # set lanetype to car
    nx.set_edge_attributes(G_lane, "M", name="lanetype")

    # set car and bike time attributes of the graph (starting from a graph with only cars)
    car_time, bike_time = {}, {}
    for u, v, k, data in G_lane.edges(data=True, keys=True):
        e = (u, v, k)
        car_time[e] = compute_car_time(data)
        bike_time[e] = compute_edgedependent_bike_time(data, shared_lane_factor=shared_lane_factor)
    nx.set_edge_attributes(G_lane, car_time, name="car_time")
    nx.set_edge_attributes(G_lane, bike_time, name="bike_time")

    # compute SP wrt car time or bike time and wrt OD matrix possibly
    betweenness, car_travel_time, bike_travel_time = compute_betweenness_and_splength(
        G_lane, betweenness_attr, od_matrix=od_matrix, sp_method=sp_method, weight_od_flow=weight_od_flow
    )
    # initialize pareto result list
    pareto_df = [
        {
            "bike_edges_added": 0,
            "bike_edges": 0,
            "car_edges": G_lane.number_of_edges(),
            "bike_time": bike_travel_time,
            "car_time": car_travel_time,
        }
    ]

    edges_removed = 0
    # max_iters = car_graph.number_of_edges() * 10
    # iteratively add edges
    while not np.all(np.array(list(is_bike_or_fixed.values()))):
        # sort betweenens centrality (use highest centrality if bike_time, so reverse)
        sorted_by_betweenness = sorted(betweenness.items(), key=lambda x: x[1], reverse=betweenness_attr == "bike_time")
        for s in sorted_by_betweenness:
            if not is_bike_or_fixed[s[0]]:
                edge_to_transform = s[0]
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
        G_lane.edges[edge_to_transform]["lanetype"] = "P"
        G_lane.edges[edge_to_transform]["car_time"] = np.inf
        new_bike_time = compute_edgedependent_bike_time(
            G_lane.edges[edge_to_transform], shared_lane_factor=shared_lane_factor
        )
        # debugging:
        # assert new_bike_time == G_lane.edges[edge_to_transform]["bike_time"] / shared_lane_factor
        G_lane.edges[edge_to_transform]["bike_time"] = new_bike_time

        # compute new travel times
        betweenness, car_travel_time, bike_travel_time = compute_betweenness_and_splength(
            G_lane, betweenness_attr, od_matrix=od_matrix, sp_method=sp_method, weight_od_flow=weight_od_flow
        )
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
