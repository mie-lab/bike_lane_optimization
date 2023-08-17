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


def ceiled_car_graph_simple(result_df):
    """
    Deprecated! - did not consider special case where car lanes can increase over the capacity
    Initial car graph is one with all the car capacity values rounded up
    """
    df = result_df.copy()
    df = edge_to_source_target(df)
    # round the number of car edges for now
    df["number_edges"] = np.ceil(df["u_c(e)"])
    # get edge key:
    df = repeat_and_edgekey(df)
    # construct graph
    G = nx.from_pandas_edgelist(df, source="source", target="target", create_using=nx.MultiDiGraph, edge_key="edge_key")
    return G


def ceiled_car_graph(result_df):
    def decrease_problematic(row):
        diff = (row["u_c(e)"] + row["u_c(e)_reversed"]) - row["capacity"]
        change_var = "u_c(e)" if row["u_c(e)"] > row["u_c(e)_reversed"] else "u_c(e)_reversed"
        row[change_var] -= diff
        return row

    # transform into dataframe of undirected edges
    street_df = result_to_streets(result_df.copy()).reset_index()
    # ceil
    street_df["u_c(e)"] = np.ceil(street_df["u_c(e)"])
    street_df["u_c(e)_reversed"] = np.ceil(street_df["u_c(e)_reversed"])

    # identify problematic rows and fix them
    row_is_problem = street_df["u_c(e)"] + street_df["u_c(e)_reversed"] > street_df["capacity"]
    fixed_rows = street_df[row_is_problem].apply(decrease_problematic, axis=1)
    fixed_street_df = pd.concat([fixed_rows, street_df[~row_is_problem]])
    # add source and target columns
    fixed_street_df = edge_to_source_target(fixed_street_df)
    # transform again into directed edge dataframe
    uc = fixed_street_df[["u_c(e)", "source", "target"]].rename({"u_c(e)": "number_edges"}, axis=1)
    uc_reversed = fixed_street_df[["u_c(e)_reversed", "source", "target"]].rename(
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


def initialize_bike_graph(result_df):
    """Initial bike graph is one with just the bike edges that are feasible given that car edges are rounded up"""
    # transform from (Edge_dir, u_b, u_c, capacity) into (Edge_undir, u_b, u_c, u_c_reversed capacity)
    street_df = result_to_streets(result_df.copy()).reset_index()
    street_df["car_cap_rounded"] = np.ceil(street_df["u_c(e)"]) + np.ceil(street_df["u_c(e)_reversed"])
    # subtract current car edges (rounded) from the overall capacity
    street_df["number_edges"] = street_df["capacity"] - street_df["car_cap_rounded"]
    # initialize bike edges where possible
    bike_df = street_df[street_df["number_edges"] > 0]
    bike_df = repeat_and_edgekey(bike_df)
    bike_df = edge_to_source_target(bike_df)
    # construct graph
    G_bike = nx.from_pandas_edgelist(
        bike_df, source="source", target="target", create_using=nx.MultiGraph, edge_key="edge_key"
    )
    return G_bike


def iteratively_redistribute_edges(car_G, bike_G, unique_edges, stop_ub_zero=True, bike_edges_to_add=None):
    """Main algorithm to assign edges"""

    def remove_uc_edge():
        car_G.remove_edge(*edge)
        if not nx.is_strongly_connected(car_G):
            car_G.add_edge(*edge)
            return False
        else:
            # print("Removed", edge)
            return True

    def remove_reversed_edge():
        car_G.remove_edge(edge[1], edge[0])
        if not nx.is_strongly_connected(car_G):
            car_G.add_edge(edge[1], edge[0])
            return False
        else:
            # print("Removed", (edge[1], edge[0]))
            return True

    unique_edges.sort_values("u_b(e)", inplace=True, ascending=False)
    total_capacity = unique_edges["capacity"].sum()

    edges_added_counter = 0
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
            bike_G.add_edge(*edge)
            edges_added_counter += 1

        assert (
            bike_G.number_of_edges() + car_G.number_of_edges() == total_capacity
        ), f"Error at {edges_added_counter} with {bike_G.number_of_edges()}, {car_G.number_of_edges()}, {total_capacity}"

        # first stopping option: we added as many bike edges as we wanted to
        if bike_edges_to_add is not None and edges_added_counter >= bike_edges_to_add:
            break
        # second stopping option: we add all bike edges that are not zero capacity (when possible without disconnecting)
        if stop_ub_zero and row["u_b(e)"] == 0 and bike_edges_to_add is None:
            break
    return bike_G, car_G


def rounding_and_splitting(result_df):
    # Initial car graph is one with all the car capacity values rounded up
    car_G = ceiled_car_graph(result_df.copy())
    assert nx.is_strongly_connected(car_G)
    # Initial bike graph is one with just the bike edges that are feasible given that car edges are rounded up
    bike_G = initialize_bike_graph(result_df.copy())
    # get unique set of edges
    unique_edges = result_to_streets(result_df.copy())

    print("Start graph edges", bike_G.number_of_edges(), car_G.number_of_edges())

    bike_G, car_G = iteratively_redistribute_edges(car_G, bike_G, unique_edges, stop_ub_zero=True)
    return bike_G, car_G


def ceiled_bike_graph(result_df):
    """Generate bike graph from capacity values"""
    bike_df = result_to_streets(result_df.copy()).reset_index()
    bike_df["number_edges"] = np.ceil(bike_df["u_b(e)"])
    # initialize bike edges whenever ceiled value is greater zero
    bike_df = bike_df[bike_df["number_edges"] > 0]
    bike_df = repeat_and_edgekey(bike_df)
    bike_df = edge_to_source_target(bike_df)
    # construct graph
    G_bike = nx.from_pandas_edgelist(
        bike_df, source="source", target="target", create_using=nx.MultiGraph, edge_key="edge_key"
    )
    return G_bike


def graph_from_integer_solution(result_df):
    """
    Derive the bike and car graphs from the capacity dataframe
    capacity_df: Dataframe with columns (Edge, u_b(e), u_c(e))
    """
    car_G = ceiled_car_graph(result_df.copy())
    bike_G = ceiled_bike_graph(result_df.copy())
    return bike_G, car_G


def pareto_frontier(
    G_original, capacity_values, shared_lane_factor, return_list=False, sp_method="all_pairs", od_matrix=None
):
    """
    Round with different cutoffs and thereby compute pareto frontier
    capacity_values: pd.DataFrame
    sp_method: str, one of {all_pairs, od} - compute the all pairs shortest paths or only on the OD matrix
    """
    assert sp_method != "od" or od_matrix is not None
    G_lane = G_original.copy()
    pareto_df = []

    # Initial car graph is one with all the car capacity values rounded up
    car_G_init = ceiled_car_graph(capacity_values.copy())
    assert nx.is_strongly_connected(car_G_init)

    # Initial bike graph is one with just the bike edges that are feasible given that car edges are rounded up
    bike_G_init = initialize_bike_graph(capacity_values.copy())
    # get unique set of edges
    unique_edges = result_to_streets(capacity_values.copy())
    print("Start graph edges", bike_G_init.number_of_edges(), car_G_init.number_of_edges())

    # compute number of edges that we could redistribute (all the undirected edges that are not part of bike_G yet)
    num_edges_redistribute = len([edge for edge, _ in unique_edges.iterrows() if edge in bike_G_init.edges()])
    print("Number edges to maximally add", num_edges_redistribute)

    num_bike_edges = 0
    for bike_edges_to_add in range(num_edges_redistribute):
        # copy graphs
        car_G = car_G_init.copy()
        bike_G = bike_G_init.copy()
        # perform rounding with cutoff point
        bike_G, car_G = iteratively_redistribute_edges(
            car_G, bike_G, unique_edges, stop_ub_zero=True, bike_edges_to_add=bike_edges_to_add
        )
        # check whether we already had this number of bike edges -> finished
        if bike_G.number_of_edges() == num_bike_edges:
            print("Early stopping at iteration", bike_edges_to_add)
            break
        num_bike_edges = bike_G.number_of_edges()

        # transform the graph layout into travel times, including gradient and penalty factor for using
        # car lanes by bike
        G_lane = add_bike_and_car_time(G_lane, bike_G, car_G, shared_lane_factor)
        # measure weighted times (floyd-warshall)
        if sp_method == "od":
            bike_travel_time = od_sp(G_lane, od_matrix, weight="biketime")
            car_travel_time = od_sp(G_lane, od_matrix, weight="cartime")
        else:
            bike_travel_time = np.mean(pd.DataFrame(nx.floyd_warshall(G_lane, weight="biketime")).values)
            car_travel_time = np.mean(pd.DataFrame(nx.floyd_warshall(G_lane, weight="cartime")).values)
        pareto_df.append(
            {"bike_edges_added": bike_edges_to_add, "bike_time": bike_travel_time, "car_time": car_travel_time}
        )
    if return_list:
        return pareto_df
    return pd.DataFrame(pareto_df)
