import pandas as pd
import networkx as nx
import numpy as np
import os
from ebike_city_tools.metrics import hypervolume_indicator


def output_to_dataframe(streetIP, G, fixed_edges = pd.DataFrame()):
    # Does not output a dataframe if mathematical program ⁄infeasible
    # if opt_val == None:
    #     print("LP infeasible")
    #     return opt_val, opt_val
    assert streetIP.objective_value is not None

    capacities = nx.get_edge_attributes(G, "capacity")

    # Creates the output dataframe
    # if fixed_values.empty:
    edge_cap = []
    def retrieve_u_for_fixed_edge(edge, column) :
        row = fixed_edges.loc[fixed_edges['Edge'] == edge]
        u = row[column].values[0]
        return u

    for i, e in enumerate(G.edges):
        opt_cap_car = streetIP.vars[f"u_{e},c"].x if not streetIP.vars[f"u_{e},c"] is None else retrieve_u_for_fixed_edge(e, 'u_c(e)')
        # cap_car[list(G.edges).index(e)].x   # Gets optimal capacity value for car
        opt_cap_bike = streetIP.vars[f"u_{e},b"].x if not streetIP.vars[f"u_{e},b"] is None else retrieve_u_for_fixed_edge(e, 'u_b(e)')
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
            (s, t, edge_u, edge_v, edgetype) = m.name[2:].split(",")
            edge_u = int(edge_u[1:])
            edge_v = int(edge_v.split(")")[0][1:])
            var_values.append([m.name, s, t, edge_u, edge_v, edgetype, m.x])
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



def combine_paretos_from_path(path, name_scheme="real_pareto_optimize_od"):
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
    return res_p


def combine_pareto_frontiers(res_p):
    """
    res_p: list of pareto frontiers
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
