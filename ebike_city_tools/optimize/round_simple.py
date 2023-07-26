import networkx as nx
import pandas as pd
import numpy as np


def result_to_streets(result_df):
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


def edge_to_source_target(df):
    if type(df.iloc[0]["Edge"]) == str:
        df["Edge"] = df["Edge"].apply(eval)
    df["source"] = df["Edge"].apply(lambda x: x[0])
    df["target"] = df["Edge"].apply(lambda x: x[1])
    return df


def repeat_and_edgekey(df):
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


def initialize_car_graph(result_df):
    df = result_df.copy()
    df = edge_to_source_target(df)
    # round the number of car edges for now
    df["number_edges"] = np.ceil(df["u_c(e)"])
    # get edge key:
    df = repeat_and_edgekey(df)
    G = nx.from_pandas_edgelist(df, source="source", target="target", create_using=nx.MultiDiGraph, edge_key="edge_key")
    return G


def initialize_bike_graph(result_df):
    bike_df = result_to_streets(result_df.copy()).reset_index()
    bike_df["car_cap_rounded"] = np.ceil(bike_df["u_c(e)"]) + np.ceil(bike_df["u_c(e)_reversed"])
    bike_df["number_edges"] = bike_df["capacity"] - bike_df["car_cap_rounded"]
    bike_df = bike_df[bike_df["number_edges"] > 0]
    #     print(len(bike_df))
    bike_df = repeat_and_edgekey(bike_df)
    bike_df = edge_to_source_target(bike_df)
    #     print(len(bike_df))
    #     bike_df
    G_bike = nx.from_pandas_edgelist(
        bike_df, source="source", target="target", create_using=nx.MultiGraph, edge_key="edge_key"
    )
    return G_bike


def iteratively_redistribute_edges(car_G, bike_G, unique_edges, stop_ub_zero=True):
    def remove_uc_edge():
        car_G.remove_edge(*edge)
        if not nx.is_strongly_connected(car_G):
            car_G.add_edge(*edge)
            return False
        else:
            #             print("Removed", edge)
            return True

    def remove_reversed_edge():
        car_G.remove_edge(edge[1], edge[0])
        if not nx.is_strongly_connected(car_G):
            car_G.add_edge(edge[1], edge[0])
            return False
        else:
            #             print("Removed", (edge[1], edge[0]))
            return True

    unique_edges.sort_values("u_b(e)", inplace=True, ascending=False)
    total_capacity = unique_edges["capacity"].sum()

    for edge, row in unique_edges.iterrows():
        if edge in bike_G.edges():
            continue

        # replace the smaller one of the u_c edges (e.g., value 0.1 means that it's maybe not necessary)
        success = False
        if row["u_c(e)"] <= row["u_c(e)_reversed"] and row["u_c(e)"] > 0:
            success = remove_uc_edge()
            if not success:
                success = remove_reversed_edge()
        elif row["u_c(e)_reversed"] > 0:
            success = remove_reversed_edge()
            if not success:
                success = remove_uc_edge()

        # if it worked for either of the directions
        if success:
            #             print("Added", edge)
            bike_G.add_edge(*edge)

        assert bike_G.number_of_edges() + car_G.number_of_edges() == total_capacity

        if stop_ub_zero and row["u_b(e)"] == 0:
            break
    return bike_G, car_G


def rounding_and_splitting(result_df):
    car_G = initialize_car_graph(result_df.copy())
    assert nx.is_strongly_connected(car_G)
    bike_G = initialize_bike_graph(result_df.copy())
    unique_edges = result_to_streets(result_df.copy())

    print("Start graph edges", bike_G.number_of_edges(), car_G.number_of_edges())

    bike_G, car_G = iteratively_redistribute_edges(car_G, bike_G, unique_edges, stop_ub_zero=True)
    return bike_G, car_G
