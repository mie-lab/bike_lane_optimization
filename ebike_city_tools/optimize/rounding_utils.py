import networkx as nx
import pandas as pd
import numpy as np

from ebike_city_tools.utils import add_bike_and_car_time
from ebike_city_tools.metrics import od_sp


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

#Auxiliary function that builds a graph from a dataframe
#We assume, that the df contains columns Edge and u_b(e)
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


#Auxiliary function that builds a graph from a dataframe
#We assume, that the df contains columns Edge, u_c(e) and u_c(e)_reversed
def build_car_network_from_df(df):
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
