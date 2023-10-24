import pandas as pd
import networkx as nx
import numpy as np
import os
from ebike_city_tools.metrics import hypervolume_indicator


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
    od["trips_per_day"] = (np.random.rand(nr_routes) * 5).astype(int)
    od = od[od["s"] != od["t"]].drop_duplicates(subset=["s", "t"])

    if nodes is not None:
        # transform into the correct node names
        node_list = np.array(sorted(list(nodes)))
        as_inds = od[["s", "t"]].values
        trips_per_day = od["trips_per_day"].values  # save flow column here
        # use as index
        od = pd.DataFrame(node_list[as_inds], columns=["s", "t"])
        od["trips_per_day"] = trips_per_day
    return od


def combine_pareto_frontiers(path, name_scheme="real_pareto_optimize_od"):
    """
    Load several pareto frontiers from a path and combine them into one set of non-dominating point
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

    # find ref point
    p_concat = pd.concat(res_p)
    ref_point = np.min(p_concat[["car_time", "bike_time"]].values, axis=0)
    print(ref_point)

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
        .values
    }

    # start with best car weight and check if we should add other points:
    for i, row in p_concat.iterrows():
        min_bike_time = min([s[0] for s in solution_set])
        min_car_time = min([s[1] for s in solution_set])
        for s in list(solution_set):
            if row["bike_time"] < s[0] and row["car_time"] < s[1]:
                solution_set.remove(s)
                solution_set.add((row["bike_time"], row["car_time"], row["bike_edges"]))
            elif row["bike_time"] < min_bike_time:
                solution_set.add((row["bike_time"], row["car_time"], row["bike_edges"]))
            elif row["car_time"] < min_car_time:
                solution_set.add((row["bike_time"], row["car_time"], row["bike_edges"]))

    # output the result as a dataframe
    combined_pareto = pd.DataFrame(solution_set, columns=["bike_time", "car_time", "bike_edges"]).sort_values(
        "bike_time"
    )
    return combined_pareto
