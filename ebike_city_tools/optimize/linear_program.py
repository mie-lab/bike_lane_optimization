import networkx as nx
import pandas as pd
import numpy as np
from mip import mip, INTEGER, CONTINUOUS
from ebike_city_tools.optimize.utils import output_to_dataframe, flow_to_df


def define_IP(
    G,
    edges_bike_list=None,
    edges_car_list=None,
    fixed_values=pd.DataFrame(),
    cap_factor=1,
    only_double_bikelanes=True,
    shared_lane_variables=True,
    shared_lane_factor=2,  # twice as long bike time on shared lanes
    od_df=None,
    bike_flow_constant=1,
    car_flow_constant=1,
    integer_problem=False,
):
    """Allocates traffic lanes to the bike network or the car network by optimizing overall travel time
    and ensuring network accessibility for both modes
    Input: Graph G
    Output: Dataframe with optimal edge capacity values for each network (bike and car)
    """
    if integer_problem:
        var_type = INTEGER
    else:
        var_type = CONTINUOUS

    # edge list where at least one of the capacities (bike or car) has not been fixed
    edge_list = list(G.edges)
    if edges_bike_list is None:
        edges_bike_list = edge_list
        edges_car_list = edge_list  # TODO: remove if we change the rounding algorithm
    edges_car_bike_list = list(set(edges_bike_list) | set(edges_car_list))

    node_list = list(G.nodes)
    n = len(G.nodes)
    m = len(G.edges)
    b = len(edges_bike_list)
    c = len(edges_car_list)
    union = len(edges_car_bike_list)

    # take into account OD matrix (if None, make all-pairs OD)
    if od_df is None:
        od_flow = np.array([[i, j] for i in range(n) for j in range(n)])
        print("USING ALL-CONNECTED OD FLOW", len(od_flow))
    else:
        # for now, just extract s t columns and ignore how much flow
        od_flow = od_df[["s", "t"]].values

    print("Number of flow variables", len(od_flow), m, len(od_flow) * m)

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
        [streetIP.add_var(name=f"f_{s},{t},{e},c", lb=0, var_type=var_type) for e in range(m)] for (s, t) in od_flow
    ]
    var_f_bike = [
        [streetIP.add_var(name=f"f_{s},{t},{e},b", lb=0, var_type=var_type) for e in range(m)] for (s, t) in od_flow
    ]
    if shared_lane_variables:
        # if allowing for shared lane usage between cars and bike, set additional variables
        var_f_shared = [
            [streetIP.add_var(name=f"f_{s},{t},{e},s", lb=0, var_type=var_type) for e in range(m)] for (s, t) in od_flow
        ]
    # capacity variables
    cap_bike = [streetIP.add_var(name=f"u_{e},b", lb=0, var_type=var_type) for e in range(b)]
    cap_car = [streetIP.add_var(name=f"u_{e},c", lb=0, var_type=var_type) for e in range(c)]

    # functions to call the variables
    def f_car(od_ind, e):
        return var_f_car[od_ind][edge_list.index(e)]

    def f_bike(od_ind, e):
        return var_f_bike[od_ind][edge_list.index(e)]

    def f_shared(od_ind, e):
        return var_f_shared[od_ind][edge_list.index(e)]

    def u_b(e):
        if e in edges_bike_list:
            return cap_bike[edges_bike_list.index(e)]
        else:
            return fixed_values.loc[edge_list.index(e), "u_b(e)"]

    def u_c(e):
        if e in edges_car_list:
            return cap_car[edges_car_list.index(e)]
        else:
            return fixed_values.loc[edge_list.index(e), "u_c(e)"]

    for v in node_list:
        for od_ind, (s, t) in enumerate(od_flow):
            streetIP += f_bike(od_ind, e) >= 0
            streetIP += f_car(od_ind, e) >= 0
            streetIP += f_shared(od_ind, e) >= 0
            if s == t:
                for e in G.out_edges(v):
                    streetIP += f_bike(od_ind, e) == 0
                    streetIP += f_shared(od_ind, e) == 0
                    streetIP += f_car(od_ind, e) == 0
            elif v == s:
                if shared_lane_variables:
                    streetIP += (
                        mip.xsum(f_shared(od_ind, e) + f_bike(od_ind, e) for e in G.out_edges(v))
                        - mip.xsum(f_shared(od_ind, e) + f_bike(od_ind, e) for e in G.in_edges(v))
                        == bike_flow_constant
                    )
                else:
                    streetIP += (
                        mip.xsum(f_bike(od_ind, e) for e in G.out_edges(v))
                        - mip.xsum(f_bike(od_ind, e) for e in G.in_edges(v))
                        == bike_flow_constant
                    )
                streetIP += (
                    mip.xsum(f_car(od_ind, e) for e in G.out_edges(v))
                    - mip.xsum(f_car(od_ind, e) for e in G.in_edges(v))
                    == car_flow_constant
                )
            elif v == t:
                if shared_lane_variables:
                    streetIP += (
                        mip.xsum(f_shared(od_ind, e) + f_bike(od_ind, e) for e in G.out_edges(v))
                        - mip.xsum(f_shared(od_ind, e) + f_bike(od_ind, e) for e in G.in_edges(v))
                        == -bike_flow_constant
                    )
                else:
                    streetIP += (
                        mip.xsum(f_bike(od_ind, e) for e in G.out_edges(v))
                        - mip.xsum(f_bike(od_ind, e) for e in G.in_edges(v))
                        == -bike_flow_constant
                    )
                streetIP += (
                    mip.xsum(f_car(od_ind, e) for e in G.out_edges(v))
                    - mip.xsum(f_car(od_ind, e) for e in G.in_edges(v))
                    == -car_flow_constant
                )
            else:
                if shared_lane_variables:
                    streetIP += (
                        mip.xsum(f_shared(od_ind, e) + f_bike(od_ind, e) for e in G.out_edges(v))
                        - mip.xsum(f_shared(od_ind, e) + f_bike(od_ind, e) for e in G.in_edges(v))
                        == 0
                    )
                else:
                    streetIP += (
                        mip.xsum(f_bike(od_ind, e) for e in G.out_edges(v))
                        - mip.xsum(f_bike(od_ind, e) for e in G.in_edges(v))
                        == 0
                    )
                streetIP += (
                    mip.xsum(f_car(od_ind, e) for e in G.out_edges(v))
                    - mip.xsum(f_car(od_ind, e) for e in G.in_edges(v))
                    == 0
                )

    # # Capacity constraints - V1
    for od_ind in range(len(od_flow)):
        for e in G.edges:
            streetIP += f_bike(od_ind, e) <= u_b(e)  # still not the sum of flows
            streetIP += f_car(od_ind, e) <= u_c(e)
    # # Capacity constraints - V2 (using the sum of flows)
    # for e in G.edges:
    # streetIP += mip.xsum([f_bike(od_ind, e) for od_ind in range(len(od_flow))]) <= u_b(e)
    # streetIP += mip.xsum([f_car(od_ind, e) for od_ind in range(len(od_flow))]) <= u_c(e)

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
            mip.xsum(f_bike(od_ind, e) * bike_time[e] for od_ind in range(len(od_flow)) for e in G.edges)
            + 5 * mip.xsum(f_car(od_ind, e) * car_time[e] for od_ind in range(len(od_flow)) for e in G.edges)
            + mip.xsum(
                f_shared(od_ind, e) * bike_time[e] * shared_lane_factor  # factor how much longer it takes on car lanes
                for od_ind in range(len(od_flow))
                for e in G.edges
            )
        )
    else:
        streetIP += (
            mip.xsum(f_bike(od_ind, e) * bike_time[e] for od_ind in range(len(od_flow)) for e in G.edges)
            + mip.xsum(f_car(od_ind, e) * car_time[e] for od_ind in range(len(od_flow)) for e in G.edges)
            # # The following lines aimed to find solution with tight capacities. However, this does not work. In the end,
            # we want distribute the whole street width, so we want the larges u_b and u_c. Also, minimizing them yields
            # extremely different results
            # + 0.1 * mip.xsum([u_c(e) for e in edges_car_list])
            # + 0.1 * mip.xsum([u_b(e) for e in edges_bike_list])
        )
    return streetIP


# def optimize_lp(G):
# Optimization
