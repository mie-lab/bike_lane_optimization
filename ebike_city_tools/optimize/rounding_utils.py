import os
import networkx as nx
import pandas as pd
import numpy as np

from ebike_city_tools.metrics import od_sp, hypervolume_indicator


def result_to_streets(result_df_in):
    result_df = result_df_in.copy()
    if type(result_df.iloc[0]["Edge"]) == str:
        result_df["Edge"] = result_df["Edge"].apply(eval)
    result_df["source<target"] = result_df["Edge"].apply(lambda x: x[0] < x[1])
    result_df.set_index("Edge", inplace=True)
    # make new representation with all in one line
    reversed_edges = result_df[~result_df["source<target"]].reset_index()
    reversed_edges["Edge"] = reversed_edges["Edge"].apply(lambda x: (x[1], x[0]))
    together = pd.merge(
        result_df[result_df["source<target"]],
        reversed_edges,
        how="outer",
        left_index=True,
        right_on="Edge",
        suffixes=("", "_reversed"),
    ).set_index("Edge")
    assert all(together["capacity"] == together["capacity_reversed"])
    #     assert all(together["u_b(e)"] == together["u_b(e)_reversed"])
    assert all(together["source<target"] != together["source<target_reversed"])
    together.drop(["capacity_reversed", "source<target_reversed", "source<target"], axis=1, inplace=True)
    if all(together["u_b(e)"] == together["u_b(e)_reversed"]):
        together.drop(["u_b(e)_reversed"], axis=1, inplace=True)
    assert len(together) == 0.5 * len(result_df)
    return together


def undirected_to_directed(undirected_df):
    """Helper function that transforms a df containing undirected information on the arcs,
    i.e. a row per pair of reverse arcs, to a df containing a row per arc."""

    correct_arcs = undirected_df.copy().loc[:, ["u_b(e)", "u_c(e)", "capacity"]]
    reverse_arcs = undirected_df.loc[:, ["u_b(e)", "u_c(e)_reversed", "capacity"]].reset_index()

    reverse_arcs["Edge"] = reverse_arcs["Edge"].apply(lambda x: (x[1], x[0]))
    reverse_arcs = reverse_arcs.set_index("Edge").rename(columns={"u_c(e)_reversed": "u_c(e)"})
    return pd.concat([correct_arcs, reverse_arcs])


def edge_to_source_target(df):
    """Transforms on column called 'Edge' of a dataframe to two columns [source, target]"""
    if type(df.iloc[0]["Edge"]) == str:
        df["Edge"] = df["Edge"].apply(eval)
    df["source"] = df["Edge"].apply(lambda x: x[0])
    df["target"] = df["Edge"].apply(lambda x: x[1])
    return df


def repeat_and_edgekey(df):
    """Helper function to transform DiGraph into MultiDiGraph"""
    # repeat rows
    df.sort_values("Edge")
    df = df.reindex(df.index.repeat(df["number_edges"]))
    # assign edge key
    df["edge_key"] = 0
    df["previous_edge"] = df["Edge"].shift(1)
    for i in range(int(df["number_edges"].max())):
        df["previous_key"] = df["edge_key"].shift(1)
        df.loc[df["previous_edge"] == df["Edge"], "edge_key"] = (
            df.loc[df["previous_edge"] == df["Edge"], "previous_key"] + 1
        )
    return df


# Auxiliary function that builds a graph from a dataframe
# We assume, that the df contains columns Edge and u_b(e)
def build_bike_network_from_df(df):
    # add source and target columns
    df = edge_to_source_target(df)
    list_of_edge_numbers = df[["u_b(e)", "source", "target"]].rename({"u_b(e)": "number_edges"}, axis=1)
    # get edge keys for nx graph
    list_of_edge_numbers["Edge"] = list_of_edge_numbers.apply(lambda x: (int(x["source"]), int(x["target"])), axis=1)
    edge_df_for_graph = repeat_and_edgekey(list_of_edge_numbers)

    # construct graph
    G = nx.from_pandas_edgelist(
        edge_df_for_graph, source="source", target="target", create_using=nx.MultiDiGraph, edge_key="edge_key"
    )
    return G


def build_car_network_from_df(df):
    """
    Auxiliary function that builds a graph from a dataframe
    Arguments:
        df: pd.Dataframe with capacity values, must have columns Edge, u_c(e) and u_c(e)_reversed
    Returns:
        G: nx.MultiDiGraph, graph with one car lane edge per capacity u_c(e)
    """
    # add source and target columns
    df = edge_to_source_target(df)
    # transform again into directed edge dataframe
    uc = df[["u_c(e)", "source", "target"]].rename({"u_c(e)": "number_edges"}, axis=1)
    uc_reversed = df[["u_c(e)_reversed", "source", "target"]].rename(
        {"u_c(e)_reversed": "number_edges", "source": "target", "target": "source"}, axis=1
    )
    directed_list = pd.concat([uc, uc_reversed]).reset_index(drop=True)
    # add edge
    directed_list["Edge"] = directed_list.apply(lambda x: (int(x["source"]), int(x["target"])), axis=1)
    # get edge keys for nx graph
    edge_df_for_graph = repeat_and_edgekey(directed_list)

    # construct graph
    G = nx.from_pandas_edgelist(
        edge_df_for_graph, source="source", target="target", create_using=nx.MultiDiGraph, edge_key="edge_key"
    )
    return G


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
