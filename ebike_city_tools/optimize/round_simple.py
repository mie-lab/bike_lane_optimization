import networkx as nx
import pandas as pd
import numpy as np
from ebike_city_tools.optimize.rounding_utils import build_car_network_from_df

from ebike_city_tools.metrics import compute_travel_times
from ebike_city_tools.optimize.rounding_utils import result_to_streets, edge_to_source_target, repeat_and_edgekey


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

    # identify problematic rows and fix them -> Attention: does lead to unconnected graph sometimes
    row_is_problem = street_df["u_c(e)"] + street_df["u_c(e)_reversed"] > street_df["capacity"]
    fixed_rows = street_df[row_is_problem].apply(decrease_problematic, axis=1)
    fixed_street_df = pd.concat([fixed_rows, street_df[~row_is_problem]])
    # add source and target columns
    fixed_street_df = edge_to_source_target(fixed_street_df)

    # construct graph
    G = build_car_network_from_df(fixed_street_df)
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

    def remove_uc_edge(edge):
        car_G.remove_edge(*edge)
        if not nx.is_strongly_connected(car_G):
            car_G.add_edge(*edge)
            return False
        return True

    def remove_reversed_edge(edge):
        car_G.remove_edge(edge[1], edge[0])
        if not nx.is_strongly_connected(car_G):
            car_G.add_edge(edge[1], edge[0])
            return False
        return True

    total_capacity = unique_edges["capacity"].sum()

    edges_added_counter = 0
    for edge, row in unique_edges.iterrows():
        if edge in bike_G.edges():
            continue

        # replace the smaller one of the u_c edges (e.g., value 0.1 means that it's maybe not necessary)
        success = False
        if row["u_c(e)"] <= row["u_c(e)_reversed"] and row["u_c(e)"] > 0:
            success = remove_uc_edge(edge)
            if not success:
                success = remove_reversed_edge(edge)
        elif row["u_c(e)_reversed"] > 0:
            success = remove_reversed_edge(edge)
            if not success:
                success = remove_uc_edge(edge)

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
        if stop_ub_zero and row["u_b(e)"] == 0:
            break
    assert nx.is_strongly_connected(car_G)
    return bike_G, car_G


def rounding_and_splitting(result_df, bike_edges_to_add=None):
    # print(result_df.head)
    # Initial car graph is one with all the car capacity values rounded up
    car_G = ceiled_car_graph(result_df.copy())
    assert nx.is_strongly_connected(car_G)
    # Initial bike graph is one with just the bike edges that are feasible given that car edges are rounded up
    bike_G = initialize_bike_graph(result_df.copy())
    # get unique set of edges
    unique_edges = result_to_streets(result_df.copy())
    # sort list of edges for prioritizing
    unique_edges.sort_values("u_b(e)", inplace=True, ascending=False)

    # print("Start graph edges", bike_G.number_of_edges(), car_G.number_of_edges())

    remaining_bike_edges_to_add = (
        max([0, bike_edges_to_add - bike_G.number_of_edges()]) if bike_edges_to_add is not None else None
    )

    bike_G, car_G = iteratively_redistribute_edges(
        car_G, bike_G, unique_edges, stop_ub_zero=True, bike_edges_to_add=remaining_bike_edges_to_add
    )
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
    G_original,
    capacity_values,
    shared_lane_factor,
    return_list=False,
    sp_method="all_pairs",
    od_matrix=None,
    weight_od_flow=False,
    valid_edges_k=None,
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
    # sort list of edges
    unique_edges.sort_values("u_b(e)", inplace=True, ascending=False)
    print("Start graph edges", bike_G_init.number_of_edges(), car_G_init.number_of_edges())

    # compute number of edges that we could redistribute (all the undirected edges that are not part of bike_G yet)
    num_edges_redistribute = len([edge for edge, _ in unique_edges.iterrows() if edge not in bike_G_init.edges()])
    print("Number edges to maximally add", num_edges_redistribute)

    num_bike_edges = -1
    for bike_edges_to_add in range(num_edges_redistribute + 1):
        # copy graphs
        car_G = car_G_init.copy()
        bike_G = bike_G_init.copy()
        # perform rounding with cutoff point
        if bike_edges_to_add > 0:
            bike_G, car_G = iteratively_redistribute_edges(
                car_G, bike_G, unique_edges, stop_ub_zero=False, bike_edges_to_add=bike_edges_to_add
            )
        # check whether we already had this number of bike edges -> finished
        if bike_G.number_of_edges() == num_bike_edges and bike_edges_to_add > 1:
            print("Early stopping at iteration", bike_edges_to_add)
            break
        num_bike_edges = bike_G.number_of_edges()

        # Compute travel times
        travel_time_dict = compute_travel_times(
            G_lane,
            bike_G,
            car_G,
            od_matrix=od_matrix,
            sp_method=sp_method,
            shared_lane_factor=shared_lane_factor,
            weight_od_flow=weight_od_flow,
        )
        travel_time_dict["bike_edges_added"] = bike_edges_to_add
        pareto_df.append(travel_time_dict)

    if return_list:
        return pareto_df
    return pd.DataFrame(pareto_df)
