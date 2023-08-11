import math
import pandas as pd

from ebike_city_tools.optimize.linear_program import define_IP
from ebike_city_tools.optimize.utils import output_to_dataframe


def round_iteratively(G):
    """This is Aurelien's method - needs to be adapted to the new code and can be made more efficient"""
    # first round:
    new_ip = define_IP(G)
    new_ip.optimize()
    dataframe_edge_cap = output_to_dataframe(new_ip, G)
    opt_val = new_ip.objective_value
    print("Init opt value", opt_val)
    # now start to fix edges
    fixed_edges_bike = []
    fixed_edges_car = []

    while opt_val is not None:
        i = 0.1
        while i < 0.5:
            # identify and round non-integer capacities
            for row_int, edge_row in dataframe_edge_cap.iterrows():
                if (math.fmod(edge_row["u_b(e)"], 1) != 0) and (
                    (math.fmod(edge_row["u_b(e)"], 1) < i) or (math.fmod(edge_row["u_b(e)"], 1) > (1 - i))
                ):
                    dataframe_edge_cap.loc[row_int, "u_b(e)"] = round(edge_row["u_b(e)"])
                    fixed_edges_bike.append(row_int)
                    # print("bike", fixed_edges_bike)
                if (math.fmod(edge_row["u_c(e)"], 1) != 0) and (
                    (math.fmod(edge_row["u_c(e)"], 1) < i) or (math.fmod(edge_row["u_c(e)"], 1) > (1 - i))
                ):
                    dataframe_edge_cap.loc[row_int, "u_c(e)"] = round(edge_row["u_c(e)"])
                    fixed_edges_car.append(row_int)
                    # print("car", fixed_edges_car)
            # Create new edge lists to reoptimse
            edge_new_car = list()
            edge_new_bike = list()
            for j in range(len(G.edges)):
                if (j in fixed_edges_car) == False:
                    edge_new_car.append(list(G.edges)[j])
                if (j in fixed_edges_bike) == False:
                    edge_new_bike.append(list(G.edges)[j])

            # Reoptimisation
            ip = define_IP(G, edge_new_bike, edge_new_car, dataframe_edge_cap)
            ip.optimize()
            dataframe_edge_cap = output_to_dataframe(ip, G)
            opt_val = ip.objective_value
            i += 0.1
            print(i, opt_val)

        ### Second part for half-integer edges
        print("Second part")
        break_variable = 1
        for row_int, edge_row in dataframe_edge_cap.iterrows():
            if math.fmod(edge_row["u_b(e)"], 1) != 0:
                dataframe_edge_cap.loc[row_int, "u_b(e)"] = round(edge_row["u_b(e)"])
                fixed_edges_bike.append(row_int)
                #           print("bike", fixed_edges_bike)
                break_variable = 0
                break
            if math.fmod(edge_row["u_c(e)"], 1) != 0:
                dataframe_edge_cap.loc[row_int, "u_c(e)"] = round(edge_row["u_c(e)"])
                fixed_edges_car.append(row_int)
                #          print("car", fixed_edges_car)
                break_variable = 0
                break
        if break_variable == 1:
            break

        # Create new edge lists to reoptimse
        edge_new_car = list()
        for i in range(len(G.edges)):
            if (i in fixed_edges_car) == False:
                edge_new_car.append(list(G.edges)[j])
        edge_new_bike = list()
        for i in range(len(G.edges)):
            if (i in fixed_edges_bike) == False:
                edge_new_bike.append(list(G.edges)[j])

        # Reoptimisation
        new_ip = define_IP(G, edge_new_bike, edge_new_car, dataframe_edge_cap)
        new_ip.optimize()
        dataframe_edge_cap = output_to_dataframe(new_ip, G)
        opt_val = new_ip.objective_value

    listoflists = []

    for s in range(len(G.nodes)):
        for t in range(len(G.nodes)):
            for e in G.edges:
                opt_flow_bike = new_ip.vars[f"f_{s},{t},{e},b"].x
                opt_flow_car = var_f_car[s][t][list(G.edges).index(e)].x
                listoflists.append(
                    [f"f_{s},{t},{e},b", f"{opt_flow_bike:.2f}", f"f_{s},{t},{e},c", f"{opt_flow_car:.2f}"]
                )
    dataframe = pd.DataFrame(data=listoflists)
    return dataframe


if __name__ == "__main__":
    from ebike_city_tools.random_graph import base_graph_with_capacity

    G = base_graph_with_capacity(20)

    round_iteratively(G)
