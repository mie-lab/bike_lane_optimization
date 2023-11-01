import os
import networkx as nx
import numpy as np
import pandas as pd
from collections import defaultdict
from ebike_city_tools.metrics import hypervolume_indicator


def output_to_dataframe(streetIP, G: nx.DiGraph, fixed_edges: pd.DataFrame = pd.DataFrame()) -> pd.DataFrame:
    """
    Convert the solution of the LP / IP into a dataframe with the optimal capacities
    Arguments:
        streetIP: mip.Model
        G: nx.DiGraph, street graph (same as used for the LP)
        fixed_edges: dataframe with fixed capacities that were not optimized
    Returns:
        capacities: pd.DataFrame with columns ("Edge", "u_b(e)", "u_c(e)", "capacity")
    """
    # Does not output a dataframe if mathematical program ⁄infeasible
    assert streetIP.objective_value is not None

    capacities = nx.get_edge_attributes(G, "capacity")

    # Creates the output dataframe
    # if fixed_values.empty:
    edge_cap = []

    def retrieve_u_for_fixed_edge(edge, column):
        row = fixed_edges.loc[fixed_edges["Edge"] == edge]
        u = row[column].values[0]
        return u

    for i, e in enumerate(G.edges):
        opt_cap_car = (
            streetIP.vars[f"u_{e},c"].x
            if not streetIP.vars[f"u_{e},c"] is None
            else retrieve_u_for_fixed_edge(e, "u_c(e)")
        )
        opt_cap_bike = (
            streetIP.vars[f"u_{e},b"].x
            if not streetIP.vars[f"u_{e},b"] is None
            else retrieve_u_for_fixed_edge(e, "u_b(e)")
        )
        edge_cap.append([e, opt_cap_bike, opt_cap_car, capacities[e]])
    dataframe_edge_cap = pd.DataFrame(data=edge_cap)
    dataframe_edge_cap.columns = ["Edge", "u_b(e)", "u_c(e)", "capacity"]
    return dataframe_edge_cap


def flow_to_df(streetIP):
    """
    Output all optimized flow variables in a dataframe
    Arguments:
        streetIP: mip.Model
    Returns:
        flow_df: pd.DataFrame with optimal flow for each (s, t, e, lanetype) combination
    """
    var_values = []
    for m in streetIP.vars:
        if m.name.startswith("f_"):
            (s, t, edge_u, edge_v, edgetype) = m.name[2:].split(",")
            edge_u = int(edge_u[1:])
            edge_v = int(edge_v.split(")")[0][1:])
            var_values.append([m.name, s, t, edge_u, edge_v, edgetype, m.x])
    flow_df = pd.DataFrame(var_values, columns=["name", "s", "t", "edge_u", "edge_v", "var_type", "flow"])
    return flow_df


def result_to_streets(capacity_df):
    """
    Convert a capacity dataframe with one row per lane into a dataframe with one row per street
    Arguments:
        capacity_df: pd.DataFrame with columns ("Edge", "u_b(e)", "u_c(e)", "capacity")
    Returns:
        street_capacity_df: pd.DataFrame with columns ("Edge", "u_b(e)", "u_c(e)", "u_c(e)_reversed", "capacity")
    """
    # if edge column is a string, we need to apply eval to convert it into a tuple
    if type(capacity_df.iloc[0]["Edge"]) == str:
        capacity_df["Edge"] = capacity_df["Edge"].apply(eval)
    capacity_df["source<target"] = capacity_df["Edge"].apply(lambda x: x[0] < x[1])
    capacity_df.set_index("Edge", inplace=True)
    # make new representation with all in one line
    reversed_edges = capacity_df[~capacity_df["source<target"]].reset_index()
    reversed_edges["Edge"] = reversed_edges["Edge"].apply(lambda x: (x[1], x[0]))
    together = pd.merge(
        capacity_df[capacity_df["source<target"]],
        reversed_edges,
        how="outer",
        left_index=True,
        right_on="Edge",
        suffixes=("", "_reversed"),
    ).set_index("Edge")
    assert all(together["capacity"] == together["capacity_reversed"])
    assert all(together["u_b(e)"] == together["u_b(e)_reversed"])
    assert all(together["source<target"] != together["source<target_reversed"])
    together.drop(
        ["capacity_reversed", "u_b(e)_reversed", "source<target_reversed", "source<target"], axis=1, inplace=True
    )
    assert len(together) == 0.5 * len(capacity_df)
    return together


def combine_paretos_from_path(path: str, name_scheme: str = "real_pareto_optimize_od") -> list:
    """
    Load several pareto frontiers from a path and combine them into one set of non-dominating point
    Arguments:
        path: str
    Returns:
        res_p: list of pd.DataFrame object, where each dataframe is one pareto frontiers
    """
    # read all files
    res_p = []
    for f in os.listdir(path):
        if name_scheme not in f:
            continue
        elif "integer" in f:
            p = pd.read_csv(os.path.join(path, f))
            res_p.append(p)
            continue
        car_weight = float(f.split("_")[-1][:-4])
        fp = os.path.join(path, f)
        p = pd.read_csv(fp)
        p["car_weight"] = car_weight
        res_p.append(p)
    return res_p


def combine_pareto_frontiers(res_p: list) -> pd.DataFrame:
    """
    Combine several pareto frontiers into one set of all non-dominating solutions
    Arguments:
        res_p: list of pd.DataFrame object, where each dataframe is one pareto frontiers
    Returns:
        combined_pareto: pd.DataFrame, dataframe with columns ["bike_time", "car_time", "bike_edges"]
    """
    # find ref point
    p_concat = pd.concat(res_p).sort_values(["bike_time", "car_time"])
    ref_point = np.max(p_concat[["car_time", "bike_time"]].values, axis=0)

    # find the best curve
    min_hi = np.inf
    best_carweight = 0
    for p in res_p:
        if p["car_weight"].nunique() > 1:
            continue
        hi = hypervolume_indicator(p[["car_time", "bike_time"]].values, ref_point=ref_point)  # np.array([1, 2]))
        #     print(car_weight)
        #     print(p["car_weight"].unique()[0], round(hi, 3))
        if hi < min_hi:
            min_hi = hi
            best_carweight = p["car_weight"].unique()[0]

    # start with the set of solutions
    solution_set = {
        tuple(e)
        for e in p_concat[p_concat["car_weight"] == best_carweight]
        .dropna()[["bike_time", "car_time", "bike_edges"]]
        .drop_duplicates(subset=["bike_time", "car_time"])
        .values
    }

    # start with best car weight and check if we should add other points:
    for i, row in p_concat.iterrows():
        # check if any point in the current solution set dominates over the point, if yes, don't add it
        is_dominated = False
        for s in list(solution_set):
            if s[0] < row["bike_time"] and s[1] < row["car_time"]:
                is_dominated = True
                break
        if not is_dominated:
            solution_set.add((row["bike_time"], row["car_time"], row["bike_edges"]))

    # check again if any of the new points dominates any other
    for s1 in list(solution_set):
        for s2 in list(solution_set):
            if s1[0] < s2[0] and s1[1] < s2[1]:
                solution_set.remove(s2)

    # output the result as a dataframe
    combined_pareto = pd.DataFrame(solution_set, columns=["bike_time", "car_time", "bike_edges"]).sort_values(
        "bike_time"
    )
    return combined_pareto.drop_duplicates(subset=["bike_time", "car_time"])


def lossless_to_undirected(graph):
    """
    Convert a directed multi graph into an undirected multigraph without loosing any edges
    Note: the networkx to_undirected() method converts each pair of reciprocal edges into a single undirected edge
    """
    edges = list(graph.edges(data=True))
    graph_unwo_edges = graph.copy()
    edges_to_remove = list(graph_unwo_edges.edges())
    graph_unwo_edges.remove_edges_from(edges_to_remove)
    graph_undir = nx.MultiGraph(graph_unwo_edges)
    graph_undir.add_edges_from(edges)
    return graph_undir


def lane_to_street_graph(g_lane):
    # transform into an undirected street graph
    g_street = nx.Graph(g_lane)
    # aggregate capacities (they should be the same)
    # caps = nx.get_edge_attributes(g_lane, "capacity")  # TODO
    caps_g_street = defaultdict(int)
    # iterate over the original edges and set capacities (same for forward and backward edge)
    for u, v, d in g_lane.edges(data=True):
        if u < v:
            caps_g_street[(u, v)] += d["capacity"]  # add the capacity of this lane
        else:
            caps_g_street[(v, u)] += d["capacity"]
    nx.set_edge_attributes(g_street, caps_g_street, "capacity")

    # now make it directed -> will automatically replace every edge by two
    g_street = g_street.to_directed()

    # get original gradients
    for u, v in g_street.edges():
        if (u, v) in g_lane.edges():
            g_street[u][v]["gradient"] = g_lane[u][v][list(g_lane[u][v].keys())[0]]["gradient"]
        # only the edge in the opposite direction existed in the original graph
        elif (v, u) in g_lane.edges():
            g_street[u][v]["gradient"] = -1 * g_lane[v][u][list(g_lane[v][u].keys())[0]]["gradient"]
        else:
            raise RuntimeError(
                "This should not happen, all edges in g_street should be in g_lane in one dir or the other"
            )
    assert nx.is_strongly_connected(g_street)
    return g_street


def deprecated_lane_to_street_graph(G_lane):
    # convert to dataframe
    G_dataframe = nx.to_pandas_edgelist(G_lane)
    G_dataframe = G_dataframe[["source", "target", "capacity", "gradient", "distance", "speed_limit"]]
    # remove loops
    G_dataframe = G_dataframe[G_dataframe["source"] != G_dataframe["target"]]
    # make undirected graph
    G_dataframe["source undir"] = G_dataframe[["source", "target"]].min(axis=1)
    G_dataframe["target undir"] = G_dataframe[["source", "target"]].max(axis=1)
    # G_dataframe.apply(lambda x: tuple(sorted([x["source"], x["target"]])), axis=1)
    # aggregate to get undirected
    undirected_edges = G_dataframe.groupby(["source undir", "target undir"]).agg(
        {"capacity": "sum", "distance": "first", "speed_limit": "first"}
    )

    # generate gradient first only for the ones where source < target
    grad_per_dir_edge = G_dataframe.groupby(["source", "target"])["gradient"].first().reset_index()
    grad_per_dir_edge = grad_per_dir_edge[grad_per_dir_edge["target"] > grad_per_dir_edge["source"]]
    undirected_edges["gradient"] = grad_per_dir_edge.set_index(["source", "target"]).to_dict()["gradient"]

    # make the same edges in the reverse direction -> gradient * (-1)
    reverse_edges = undirected_edges.reset_index()
    reverse_edges["source"] = reverse_edges["target undir"]
    reverse_edges["target"] = reverse_edges["source undir"]
    reverse_edges["gradient"] = reverse_edges["gradient"] * (-1)
    # concatenate
    directed_edges = pd.concat(
        [
            undirected_edges.reset_index().rename({"source undir": "source", "target undir": "target"}, axis=1),
            reverse_edges.drop(["source undir", "target undir"], axis=1),
        ]
    ).reset_index(drop=True)

    G_streets = nx.from_pandas_edgelist(
        directed_edges,
        source="source",
        target="target",
        edge_attr=["capacity", "distance", "gradient", "speed_limit"],
        create_using=nx.DiGraph,
    )
    # set attributes (not really needed)
    # attrs = {i: {"loc": coords[i, :2], "elevation": coords[i, 2]} for i in range(len(coords))}
    # nx.set_node_attributes(G_streets, attrs)
    return G_streets


def extend_od_circular(od, nodes):
    """
    Create new OD matrix that ensures connectivity by connecting one node to the next in a list
    od: pd.DataFrame, original OD with columns s, t and trips_per_day
    nodes: list, all nodes in the graph
    """
    # shuffle and convert to df
    new_od_paths = pd.DataFrame(nodes, columns=["s"]).sample(frac=1).reset_index(drop=True)
    # shift by one to ensure cirularity
    new_od_paths["t"] = new_od_paths["s"].shift(1)
    # fille nan
    new_od_paths.loc[0, "t"] = new_od_paths.iloc[-1]["s"]

    # concatenate and add flow of 0 to the new OD pairs
    od_new = pd.concat([od, new_od_paths]).fillna(0).astype(int)

    # make sure that no loops
    od_new = od_new[od_new["s"] != od_new["t"]]
    return od_new.drop_duplicates()


def extend_od_matrix(od, nodes):
    """
    Extend the OD matrix such that every node appears as s and every node appears as t
    od: initial OD matrix, represented as a pd.Dataframe
    nodes: list of nodes in the graph
    """

    def get_missing_nodes(od_df):
        nodes_not_in_s = [n for n in nodes if n not in od_df["s"].values]
        nodes_not_in_t = [n for n in nodes if n not in od_df["t"].values]
        return nodes_not_in_s, nodes_not_in_t

    # find missing nodes
    nodes_not_in_s, nodes_not_in_t = get_missing_nodes(od)
    min_len = min([len(nodes_not_in_s), len(nodes_not_in_t)])
    len_diff = max([len(nodes_not_in_s), len(nodes_not_in_t)]) - min_len
    # combine every node of the longer list with a random permutation of the smaller list and add up
    if min_len == len(nodes_not_in_t):
        shuffled_t = np.random.permutation(nodes_not_in_t)
        combined_nodes = np.concatenate(
            [
                np.stack([nodes_not_in_s[:min_len], shuffled_t]),
                np.stack([nodes_not_in_s[min_len:], shuffled_t[:len_diff]]),
            ],
            axis=1,
        )
    else:
        shuffled_s = np.random.permutation(nodes_not_in_s)
        combined_nodes = np.concatenate(
            [
                np.stack([shuffled_s, nodes_not_in_t[:min_len]]),
                np.stack([shuffled_s[:len_diff], nodes_not_in_t[min_len:]]),
            ],
            axis=1,
        )
    # transform to dataframe
    new_od_paths = pd.DataFrame(combined_nodes.swapaxes(1, 0), columns=["s", "t"])
    # concat and add a flow value of 0 since we don't want to optimize the travel time for these lanes
    od_new = pd.concat([od, new_od_paths]).fillna(0)

    # check again
    nodes_not_in_s, nodes_not_in_t = get_missing_nodes(od_new)
    assert len(nodes_not_in_s) == 0 and len(nodes_not_in_t) == 0
    return od_new


def compute_bike_time(distance, gradient):
    """
    Following the formula from Parkin and Rotheram (2010)
    """
    if gradient > 0:
        # gradient positive -> reduced speed
        return distance / max([21.6 - 1.44 * gradient, 1])  # at least 1kmh speed
    else:
        # gradient positive -> increase speed (but - because gradient also negative)
        return distance / (21.6 - 0.86 * gradient)


def add_bike_and_car_time(G_lane, bike_G, car_G, shared_lane_factor=2):
    """
    DEPRECATED --> use function below (output_lane_graph)
    Add two attributes to the graph G_lane: biketime and cartime
    """
    bike_edges = bike_G.edges()
    car_edges = car_G.edges()

    # thought: can keep it a multigraph because we are iterating
    for u, v, k, attributes in G_lane.edges(data=True, keys=True):
        e = (u, v, k)

        # 1) Car travel time: simply check whether the edge exists:
        if (u, v) in car_edges:
            G_lane.edges[e]["cartime"] = 60 * attributes["distance"] / attributes["speed_limit"]
        else:
            G_lane.edges[e]["cartime"] = np.inf

        # 2) Bike travel time
        # Case 1: There is a bike lane on this edge
        if (u, v) in bike_edges:
            G_lane.edges[e]["biketime"] = 60 * compute_bike_time(attributes["distance"], attributes["gradient"])
        # Case 2: There is no bike lane, but a car lane in the right direction --> multiply with shared_lane_factor
        elif (u, v) in car_edges:
            G_lane.edges[e]["biketime"] = (
                60 * shared_lane_factor * compute_bike_time(attributes["distance"], attributes["gradient"])
            )
        # Case 3: Neither a bike lane nor a directed car lane exists
        else:
            G_lane.edges[e]["biketime"] = np.inf
    return G_lane


def compute_car_time(row):
    if row["lanetype"] == "M":
        return 60 * row["distance"] / row["speed_limit"]
    else:
        return np.inf


def compute_edgedependent_bike_time(row, shared_lane_factor: int = 2):
    """Following the formula from Parkin and Rotheram (2010)"""
    biketime = 60 * compute_bike_time(row["distance"], row["gradient"])
    if row["lanetype"] == "P":
        return biketime
    else:
        return biketime * shared_lane_factor


def compute_penalized_car_time(attr_dict: dict, bike_lane_speed: int = 10) -> int:
    if attr_dict["lanetype"] == "M":
        return 60 * attr_dict["distance"] / attr_dict["speed_limit"]
    elif attr_dict["lanetype"] == "P":
        # return np.inf  # uncomment to test top-down approach with infinite paths
        return 60 * attr_dict["distance"] / bike_lane_speed  # assume speed limit on bike priority lanes
    else:
        raise RuntimeError("lanetyp other than M and P not implemented")


def output_lane_graph(
    G_lane,
    bike_G,
    car_G,
    shared_lane_factor=2,
    output_attr=["width", "distance", "length", "speed_limit", "fixed", "gradient"],
):
    """
    Output a lane graph in the format of the SNMan standard.
    Arguments:
        G_lane: Input lane graph (original street network) -> this is used to get the edge attributes
        car_G: nx.MultiDiGraph, Output graph of car network
        bike_G: nx.MultiGraph, Output graph of bike network
    """

    assert bike_G.number_of_edges() + car_G.number_of_edges() == G_lane.number_of_edges()

    # Step 1: make one dataframe of all car and bike edges
    car_edges = nx.to_pandas_edgelist(car_G)
    car_edges["lane"] = "M>"
    car_edges["lanetype"] = "M"
    bike_edges = nx.to_pandas_edgelist(bike_G.to_directed())
    bike_edges["lane"] = "P>"
    bike_edges["lanetype"] = "P"
    all_edges = pd.concat([car_edges, bike_edges])
    all_edges["direction"] = ">"
    all_edges.drop(output_attr, axis=1, inplace=True, errors="ignore")

    # Step 2: get all attributes of the original edges (in both directions)
    edges_G_lane = nx.to_pandas_edgelist(G_lane)
    # revert edges
    edges_G_lane_reversed = edges_G_lane.copy()
    edges_G_lane_reversed["gradient"] *= -1
    edges_G_lane_reversed["source_temp"] = edges_G_lane_reversed["source"]
    edges_G_lane_reversed["source"] = edges_G_lane_reversed["target"]
    edges_G_lane_reversed["target"] = edges_G_lane_reversed["source_temp"]
    # concat
    edges_G_lane_doubled = pd.concat([edges_G_lane, edges_G_lane_reversed])
    # extract relevant attributes --> these are all attributes that we can merge with the others
    agg_dict = {attr: "first" for attr in output_attr if attr in edges_G_lane_doubled.columns}
    edges_G_lane_doubled = edges_G_lane_doubled.groupby(["source", "target"]).agg(agg_dict)

    # Step 3: Merge with the attributes
    all_edges_with_attributes = all_edges.merge(
        edges_G_lane_doubled, how="left", left_on=["source", "target"], right_on=["source", "target"]
    )
    # Step 4: compute bike and car time
    all_edges_with_attributes["cartime"] = all_edges_with_attributes.apply(compute_car_time, axis=1)
    all_edges_with_attributes["biketime"] = all_edges_with_attributes.apply(
        compute_edgedependent_bike_time, shared_lane_factor=shared_lane_factor, axis=1
    )

    # Step 5: make a graph
    attrs = [c for c in all_edges_with_attributes.columns if c not in ["source", "target"]]
    G_final = nx.from_pandas_edgelist(
        all_edges_with_attributes, edge_attr=attrs, source="source", target="target", create_using=nx.MultiDiGraph
    )
    # make sure that the other attributes are transferred to the new graph
    G_final = transfer_node_attributes(G_lane, G_final)
    return G_final


def filter_by_attribute(G, attr, attr_value):
    """
    Create a subgraph of G consisting of the edges with attr=attr_value
    """
    edges_df = nx.to_pandas_edgelist(G)
    edges_df = edges_df[edges_df[attr] == attr_value]
    attr_cols = [c for c in edges_df.columns if c not in ["source", "target"]]
    G_filtered = nx.from_pandas_edgelist(
        edges_df, edge_attr=attr_cols, source="source", target="target", create_using=type(G)
    )
    # make sure that the other attributes are transferred to the new graph
    G_filtered = transfer_node_attributes(G, G_filtered)
    return G_filtered


def transfer_node_attributes(G_in, G_out):
    for n, data in G_in.nodes(data=True):
        attr_keys = data.keys()
        break
    for a in attr_keys:
        attr_dict = nx.get_node_attributes(G_in, a)
        nx.set_node_attributes(G_out, attr_dict, name=a)
    # transfer graph
    for k in G_in.graph.keys():
        G_out.graph[k] = G_in.graph[k]
    return G_out
