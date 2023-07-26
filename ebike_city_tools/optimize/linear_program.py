import networkx as nx
import pandas as pd
from mip import mip
from ebike_city_tools.optimize.utils import output_to_dataframe


def initialize_IP(
    G,
    edges_bike_list=None,
    edges_car_list=None,
    fixed_values=pd.DataFrame(),
    cap_factor=2,
    only_double_bikelanes=True,
    shared_lane_variables=True,
    shared_lane_factor=2,  # twice as long bike time on shared lanes
):
    """Allocates traffic lanes to the bike network or the car network by optimizing overall travel time
    and ensuring network accessibility for both modes
    Input: Graph G
    Output: Dataframe with optimal edge capacity values for each network (bike and car)
    """
    print("CAP FACTOR", cap_factor)

    # edge list where at least one of the capacities (bike or car) has not been fixed
    if edges_bike_list is None:
        edges_bike_list = list(G.edges)
        edges_car_list = list(G.edges)  # TODO: remove if we change the rounding algorithm
    edges_car_bike_list = list(set(edges_bike_list) | set(edges_car_list))

    n = len(G.nodes)
    m = len(G.edges)
    b = len(edges_bike_list)
    c = len(edges_car_list)
    union = len(edges_car_bike_list)

    capacities = nx.get_edge_attributes(G, "capacity")
    distance = nx.get_edge_attributes(G, "distance")
    gradient = nx.get_edge_attributes(G, "gradient")

    # Creation of time attribute
    for e in G.edges:
        if gradient[e] > 0:
            G.edges[e]["biketime"] = distance[e] / max([21.6 - 1.44 * gradient[e], 1])  # at least 1kmh speed
        else:  # if gradient < 0, than speed must increase --> - * -
            G.edges[e]["biketime"] = distance[e] / (21.6 - 0.86 * gradient[e])
        G.edges[e]["cartime"] = distance[e] / 30
    bike_time = nx.get_edge_attributes(G, "biketime")
    car_time = nx.get_edge_attributes(G, "cartime")

    # Optimization problem
    streetIP = mip.Model(name="bike lane allocation", sense=mip.MINIMIZE)

    # variables

    # flow variables

    var_f_car = [
        [[streetIP.add_var(name=f"f_{s},{t},{e},c", lb=0) for e in range(m)] for t in range(n)] for s in range(n)
    ]
    var_f_bike = [
        [[streetIP.add_var(name=f"f_{s},{t},{e},b", lb=0) for e in range(m)] for t in range(n)] for s in range(n)
    ]
    if shared_lane_variables:
        # if allowing for shared lane usage between cars and bike, set additional variables
        var_f_shared = [
            [[streetIP.add_var(name=f"f_{s},{t},{e},s", lb=0) for e in range(m)] for t in range(n)] for s in range(n)
        ]
    # capacity variables
    cap_bike = [streetIP.add_var(name=f"u_{e},b", lb=0) for e in range(b)]
    cap_car = [streetIP.add_var(name=f"u_{e},c", lb=0) for e in range(c)]

    # functions to call the variables
    def f_car(s, t, e):
        return var_f_car[s][t][list(G.edges).index(e)]

    def f_bike(s, t, e):
        return var_f_bike[s][t][list(G.edges).index(e)]

    def f_shared(s, t, e):
        return var_f_shared[s][t][list(G.edges).index(e)]

    def u_b(e):
        if e in edges_bike_list:
            return cap_bike[edges_bike_list.index(e)]
        else:
            return fixed_values.loc[list(G.edges).index(e), "u_b(e)"]

    def u_c(e):
        if e in edges_car_list:
            return cap_car[edges_car_list.index(e)]
        else:
            return fixed_values.loc[list(G.edges).index(e), "u_c(e)"]

    # Flow constraints
    bike_flow_constant = 1
    car_flow_constant = 1
    # -> currently 1 and -1
    for v in range(n):
        for s in range(n):
            for t in range(n):
                streetIP += f_bike(s, t, e) >= 0
                streetIP += f_car(s, t, e) >= 0
                streetIP += f_shared(s, t, e) >= 0
                if s == t:
                    for e in G.out_edges(v):
                        streetIP += f_bike(s, t, e) == 0
                        streetIP += f_shared(s, t, e) == 0
                        streetIP += f_car(s, t, e) == 0
                elif v == s:
                    if shared_lane_variables:
                        streetIP += (
                            mip.xsum(f_shared(s, t, e) + f_bike(s, t, e) for e in G.out_edges(v))
                            - mip.xsum(f_shared(s, t, e) + f_bike(s, t, e) for e in G.in_edges(v))
                            == bike_flow_constant
                        )
                    else:
                        streetIP += (
                            mip.xsum(f_bike(s, t, e) for e in G.out_edges(v))
                            - mip.xsum(f_bike(s, t, e) for e in G.in_edges(v))
                            == bike_flow_constant
                        )
                    streetIP += (
                        mip.xsum(f_car(s, t, e) for e in G.out_edges(v))
                        - mip.xsum(f_car(s, t, e) for e in G.in_edges(v))
                        == car_flow_constant
                    )
                elif v == t:
                    if shared_lane_variables:
                        streetIP += (
                            mip.xsum(f_shared(s, t, e) + f_bike(s, t, e) for e in G.out_edges(v))
                            - mip.xsum(f_shared(s, t, e) + f_bike(s, t, e) for e in G.in_edges(v))
                            == -bike_flow_constant
                        )
                    else:
                        streetIP += (
                            mip.xsum(f_bike(s, t, e) for e in G.out_edges(v))
                            - mip.xsum(f_bike(s, t, e) for e in G.in_edges(v))
                            == -bike_flow_constant
                        )
                    streetIP += (
                        mip.xsum(f_car(s, t, e) for e in G.out_edges(v))
                        - mip.xsum(f_car(s, t, e) for e in G.in_edges(v))
                        == -car_flow_constant
                    )
                else:
                    if shared_lane_variables:
                        streetIP += (
                            mip.xsum(f_shared(s, t, e) + f_bike(s, t, e) for e in G.out_edges(v))
                            - mip.xsum(f_shared(s, t, e) + f_bike(s, t, e) for e in G.in_edges(v))
                            == 0
                        )
                    else:
                        streetIP += (
                            mip.xsum(f_bike(s, t, e) for e in G.out_edges(v))
                            - mip.xsum(f_bike(s, t, e) for e in G.in_edges(v))
                            == 0
                        )
                    streetIP += (
                        mip.xsum(f_car(s, t, e) for e in G.out_edges(v))
                        - mip.xsum(f_car(s, t, e) for e in G.in_edges(v))
                        == 0
                    )
    # # Capacity constraints - V1
    for s in range(n):
        for t in range(n):
            for e in G.edges:
                streetIP += f_bike(s, t, e) <= u_b(e)  # still not the sum of flows
                streetIP += f_car(s, t, e) <= u_c(e)
    # # Capacity constraints - V2 (using the sum of flows)
    # for e in G.edges:
    #     streetIP += mip.xsum([f_bike(s, t, e) for s in range(n) for t in range(n)]) <= u_b(e)
    #     streetIP += mip.xsum([f_car(s, t, e) for s in range(n) for t in range(n)]) <= u_c(e)

    if only_double_bikelanes:
        for i in range(b):
            # both directions for the bike have the same capacity
            streetIP += u_b((edges_bike_list[i][0], edges_bike_list[i][1])) == u_b(
                (edges_bike_list[i][1], edges_bike_list[i][0])
            )
    for i in range(union):
        # take half - edges_car_bike_list[i]=(0,9) dann nehmen wir das (0,9) und die (9,0) capacities
        streetIP += (
            u_b((edges_car_bike_list[i][0], edges_car_bike_list[i][1])) / 2
            + u_c((edges_car_bike_list[i][0], edges_car_bike_list[i][1]))
            + u_c((edges_car_bike_list[i][1], edges_car_bike_list[i][0]))
            + u_b((edges_car_bike_list[i][1], edges_car_bike_list[i][0])) / 2
            <= capacities[(edges_car_bike_list[i][0], edges_car_bike_list[i][1])] * cap_factor
        )

    # Objective value
    if shared_lane_variables:
        streetIP += (
            mip.xsum(f_bike(s, t, e) * bike_time[e] for s in range(n) for t in range(n) for e in G.edges)
            + 5 * mip.xsum(f_car(s, t, e) * car_time[e] for s in range(n) for t in range(n) for e in G.edges)
            + mip.xsum(
                f_shared(s, t, e) * bike_time[e] * shared_lane_factor  # factor how much longer it takes on car lanes
                for s in range(n)
                for t in range(n)
                for e in G.edges
            )
        )
    else:
        streetIP += (
            mip.xsum(f_bike(s, t, e) * bike_time[e] for s in range(n) for t in range(n) for e in G.edges)
            + mip.xsum(f_car(s, t, e) * car_time[e] for s in range(n) for t in range(n) for e in G.edges)
            # # The following lines aimed to find solution with tight capacities. However, this does not work. In the end,
            # we want distribute the whole street width, so we want the larges u_b and u_c. Also, minimizing them yields
            # extremely different results
            # + 0.1 * mip.xsum([u_c(e) for e in edges_car_list])
            # + 0.1 * mip.xsum([u_b(e) for e in edges_bike_list])
        )
    return streetIP


# def optimize_lp(G):
# Optimization

if __name__ == "__main__":
    from ebike_city_tools.random_graph import city_graph, lane_to_street_graph
    import numpy as np

    np.random.seed(20)

    G_city = city_graph(20)
    G = lane_to_street_graph(G_city)

    # # for flow-sum-constraint: compute maximal number of paths that could pass through one edge
    # max_paths_one_edge = G.number_of_nodes() ** 2  # TODO: replace with length of OD matrix
    # FACTOR_MAX_PATHS = 0.1  # only half of the paths are allowed to use the same street
    # cap_factor = max_paths_one_edge * FACTOR_MAX_PATHS

    ip = initialize_IP(G, cap_factor=1)
    ip.optimize()

    dataframe_edge_cap = output_to_dataframe(ip, G)
    opt_val = ip.objective_value
    print("OPT VALUE", opt_val)
    # return dataframe_edge_cap
    # print(G_input.edges(data=True))
    # test = optimize_lp(G_input)
    dataframe_edge_cap.to_csv("outputs/test_lp_solution.csv", index=False)
