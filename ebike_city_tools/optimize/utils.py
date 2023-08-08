import pandas as pd
import networkx as nx
import numpy as np


def output_to_dataframe(streetIP, G):
    # Does not output a dataframe if mathematical program ⁄infeasible
    # if opt_val == None:
    #     print("LP infeasible")
    #     return opt_val, opt_val
    assert streetIP.objective_value is not None

    capacities = nx.get_edge_attributes(G, "capacity")

    # Creates the output dataframe
    # if fixed_values.empty:
    edge_cap = []
    for i, e in enumerate(G.edges):
        opt_cap_car = streetIP.vars[f"u_{i},c"].x
        # cap_car[list(G.edges).index(e)].x   # Gets optimal capacity value for car
        opt_cap_bike = streetIP.vars[f"u_{i},b"].x
        # cap_bike[list(G.edges).index(e)].x # Gets optimal capacity value for bike
        edge_cap.append([e, opt_cap_bike, opt_cap_car, capacities[e]])
    dataframe_edge_cap = pd.DataFrame(data=edge_cap)
    dataframe_edge_cap.columns = ["Edge", "u_b(e)", "u_c(e)", "capacity"]
    # else:
    #     for e in edges_car_list:
    #         fixed_values.loc[list(G.edges).index(e), "u_c(e)"] = u_c(e).x
    #     for e in edges_bike_list:
    #         fixed_values.loc[list(G.edges).index(e), "u_b(e)"] = u_b(e).x
    #     dataframe_edge_cap = fixed_values
    return dataframe_edge_cap


def flow_to_df(streetIP, edge_list):
    var_values = []
    for m in streetIP.vars:
        if m.name.startswith("f_"):
            (s, t, e_index, edgetype) = m.name[2:].split(",")
            edge = edge_list[int(e_index)]
            var_values.append([m.name, s, t, edge[0], edge[1], edgetype, m.x])
    dataframe = pd.DataFrame(var_values, columns=["name", "s", "t", "edge_u", "edge_v", "var_type", "flow"])
    return dataframe


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
    assert all(together["u_b(e)"] == together["u_b(e)_reversed"])
    assert all(together["source<target"] != together["source<target_reversed"])
    together.drop(
        ["capacity_reversed", "u_b(e)_reversed", "source<target_reversed", "source<target"], axis=1, inplace=True
    )
    assert len(together) == 0.5 * len(result_df)
    return together


def make_fake_od(n, nr_routes, nodes=None):
    od = pd.DataFrame()
    od["s"] = (np.random.rand(nr_routes) * n).astype(int)
    od["t"] = (np.random.rand(nr_routes) * n).astype(int)
    od["flow"] = (np.random.rand(nr_routes) * 5).astype(int)
    od = od[od["s"] != od["t"]].drop_duplicates(subset=["s", "t"])

    if nodes is not None:
        # transform into the correct node names
        node_list = np.array(sorted(list(nodes)))
        as_inds = od[["s", "t"]].values
        od = pd.DataFrame(node_list[as_inds], columns=["s", "t"])
    return od
