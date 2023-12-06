import networkx as nx
import pandas as pd
import numpy as np
from mip import mip, INTEGER, CONTINUOUS
from ebike_city_tools.utils import compute_bike_time


def define_IP(
    G,
    edges_bike_list=None,
    edges_car_list=None,
    valid_edges_per_od_pair=None,
    fixed_edges=pd.DataFrame(),
    cap_factor=1,
    only_double_bikelanes=True,
    shared_lane_variables=True,
    shared_lane_factor=2,
    od_df=None,
    bike_flow_constant=1,
    car_flow_constant=1,
    weight_od_flow=False,
    integer_problem=False,
    car_weight=5,
):
    """
    Allocates traffic lanes to the bike network or the car network by optimizing overall travel time
    and ensuring network accessibility for both modes
    Arguments:
        G: input graph (nx.DiGraph)
        edges_bike_list: list of edges to optimize, assuming that some edges are fixed already (iterative rounding)
        edges_car_list: list of edges to optimize, assuming that some edges are fixed already (iterative rounding)
        fixed_edges: edge capacities that are already fixed (iterative rounding)
        cap_factor: Factor to increase the capacity (deprecated)
        only_double_bikelanes: Allow only bidirectional bike lanes
        shared_lane_variables: If True, bikes are allowed to drive on car lanes (i.e., shared lanes) under penalty
        shared_lane_factor: penalty factor for panalizing biking on car lanes, e.g. 2 means that the travel time is
            twice as long on shared lanes than on pure bike lanes
        od_df: Dataframe with OD pairs (columns s, t and trips_per_day)
        bike_flow_constant: Required flow value to be sent from s to t by bike
        car_flow_constant: Required flow value to be sent from s to t by car
        weight_od_flow: If True, the terms in the objective are weighted by the flow in the OD matrix
        integer_problem: if True, the flow variables are constraint to be integers
        car_weight: int, weighting of car travel time in the objective function
    Returns: Dataframe with optimal edge capacity values for each network (bike and car)
    """
    if integer_problem:
        var_type = INTEGER
    else:
        var_type = CONTINUOUS

    # edge list where at least one of the capacities (bike or car) has not been fixed
    edge_list = list(G.edges)
    fixed_edge_list = []
    if fixed_edges.shape[1] > 0:
        fixed_edge_list = fixed_edges["Edge"].values.tolist()

    if edges_bike_list is None:
        edges_bike_list = list(set(edge_list) - set(fixed_edge_list))
        edges_car_list = edges_bike_list  # TODO: remove if we change the rounding algorithm
    edges_car_bike_list = list(set(edges_bike_list) | set(edges_car_list))

    node_list = list(G.nodes)
    number_nodes = len(G.nodes)
    number_edges = len(G.edges)
    union = len(edges_car_bike_list)

    # take into account OD matrix (if None, make all-pairs OD)
    if od_df is None:
        od_flow = np.array([[i, j] for i in range(number_nodes) for j in range(number_nodes)])
        print("USING ALL-CONNECTED OD FLOW", len(od_flow))
    else:
        # for now, just extract s t columns and ignore how much flow
        od_flow = od_df[["s", "t"]].values

    # If there are no arc restrictions specified, all arcs are feasible to take.
    # TODO we might want to change this, such that we do not save the edge_list multiple times.
    # this could for example be done by using a function that evaluates valid_edges_per_od_pair or else returns
    # edge_list.
    if valid_edges_per_od_pair is None:
        valid_edges_per_od_pair = {}
        for (s,t) in od_df[["s", "t"]].values:
            valid_edges_per_od_pair[(s,t)]=edge_list

    # if desired, we weight the terms in the objective function by the flow in the OD matrix
    if weight_od_flow:
        assert od_df is not None, "if weight_od_flow=True, an OD matrix must be provided!"
        od_weighting = od_df["trips_per_day"].values
    elif od_df is not None:
        # don't weight by the flow, but still keep the values for auxiliary OD-pairs at zero
        od_weighting = (od_df["trips_per_day"].values > 0).astype(int)
        # prevent them from being all zero
        if np.all(od_weighting == 0):
            od_weighting = np.ones(len(od_weighting))
    else:
        od_weighting = np.ones(len(od_flow))

    print(f"Theoretical number of flow variables: {len(od_flow) * number_edges} ({number_edges} edges and {len(od_flow)} OD pairs)")

    capacities = nx.get_edge_attributes(G, "capacity")
    distance = nx.get_edge_attributes(G, "distance")
    speed_limit = nx.get_edge_attributes(G, "speed_limit")
    gradient = nx.get_edge_attributes(G, "gradient")

    # Creation of time attribute
    for e in G.edges:
        G.edges[e]["bike_time"] = compute_bike_time(distance[e], gradient[e])
        G.edges[e]["car_time"] = distance[e] / speed_limit[e]
    bike_time = nx.get_edge_attributes(G, "bike_time")
    car_time = nx.get_edge_attributes(G, "car_time")

    # Optimization problem
    streetIP = mip.Model(name="bike lane allocation", sense=mip.MINIMIZE)

    # variables

    # flow variables

    var_f_car = [
        [streetIP.add_var(name=f"f_{s},{t},{e},c", lb=0, var_type=var_type) for e in valid_edges_per_od_pair[(s,t)]] for (s, t) in od_flow
    ]
    var_f_bike = [
        [streetIP.add_var(name=f"f_{s},{t},{e},b", lb=0, var_type=var_type) for e in valid_edges_per_od_pair[(s,t)]] for (s, t) in od_flow
    ]
    if shared_lane_variables:
        # if allowing for shared lane usage between cars and bike, set additional variables
        var_f_shared = [
            [streetIP.add_var(name=f"f_{s},{t},{e},s", lb=0, var_type=var_type) for e in valid_edges_per_od_pair[(s,t)]]
            for (s, t) in od_flow
        ]
    # capacity variables
    cap_bike = [streetIP.add_var(name=f"u_{e},b", lb=0, var_type=var_type) for e in edges_bike_list]
    cap_car = [streetIP.add_var(name=f"u_{e},c", lb=0, var_type=var_type) for e in edges_car_list]

    # functions to call the variables
    def f_car(od_ind, e):
        s = od_flow[od_ind][0]
        t = od_flow[od_ind][1]
        if e in valid_edges_per_od_pair[(s,t)]:
            return var_f_car[od_ind][valid_edges_per_od_pair[(s,t)].index(e)]
        else:
            return 0

    def f_bike(od_ind, e):
        s = od_flow[od_ind][0]
        t = od_flow[od_ind][1]
        if e in valid_edges_per_od_pair[(s,t)]:
            return var_f_bike[od_ind][valid_edges_per_od_pair[(s,t)].index(e)]
        else:
            return 0

    def f_shared(od_ind, e):
        s = od_flow[od_ind][0]
        t = od_flow[od_ind][1]
        if e in valid_edges_per_od_pair[(s,t)]:
            return var_f_shared[od_ind][valid_edges_per_od_pair[(s,t)].index(e)]
        else:
            return 0

    def u_b(e):
        if e in fixed_edge_list:
            row = fixed_edges.loc[fixed_edges["Edge"] == e]
            u_b = row["u_b(e)"].values[0]
            return u_b
        else:
            return cap_bike[edges_bike_list.index(e)]

    def u_c(e):
        if e in fixed_edge_list:
            row = fixed_edges.loc[fixed_edges["Edge"] == e]
            u_c = row["u_c(e)"].values[0]
            return u_c
        else:
            return cap_car[edges_car_list.index(e)]

    for v in node_list:
        for od_ind, (s, t) in enumerate(od_flow):

            def add_flow_constraints(streetIP, bike_excess, car_excess):
                if shared_lane_variables:
                    streetIP += (
                        mip.xsum(f_shared(od_ind, e) + f_bike(od_ind, e) for e in G.out_edges(v))
                        - mip.xsum(f_shared(od_ind, e) + f_bike(od_ind, e) for e in G.in_edges(v))
                        == bike_excess
                    )
                else:
                    streetIP += (
                        mip.xsum(f_bike(od_ind, e) for e in G.out_edges(v))
                        - mip.xsum(f_bike(od_ind, e) for e in G.in_edges(v))
                        == bike_excess
                    )
                streetIP += (
                    mip.xsum(f_car(od_ind, e) for e in G.out_edges(v))
                    - mip.xsum(f_car(od_ind, e) for e in G.in_edges(v))
                    == car_excess
                )

            if s == t:
                for e in G.out_edges(v):
                    streetIP += f_bike(od_ind, e) == 0
                    streetIP += f_shared(od_ind, e) == 0
                    streetIP += f_car(od_ind, e) == 0
            elif v == s:
                add_flow_constraints(streetIP, bike_flow_constant, car_flow_constant)
            elif v == t:
                add_flow_constraints(streetIP, -bike_flow_constant, -car_flow_constant)
            else:
                add_flow_constraints(streetIP, 0, 0)

    # # Capacity constraints - V1
    for od_ind in range(len(od_flow)):
        (s,t) = od_flow[od_ind]
        for e in valid_edges_per_od_pair[(s,t)]:
            streetIP += f_bike(od_ind, e) <= u_b(e)  # still not the sum of flows
            streetIP += f_car(od_ind, e) <= u_c(e)
    # # Capacity constraints - V2 (using the sum of flows)
    # for e in G.edges:
    # streetIP += mip.xsum([f_bike(od_ind, e) for od_ind in range(len(od_flow))]) <= u_b(e)
    # streetIP += mip.xsum([f_car(od_ind, e) for od_ind in range(len(od_flow))]) <= u_c(e)

    if only_double_bikelanes:
        for i in range(len(edges_bike_list)):
            # both directions for the bike have the same capacity
            streetIP += u_b((edges_bike_list[i][0], edges_bike_list[i][1])) == u_b(
                (edges_bike_list[i][1], edges_bike_list[i][0])
            )
    for i in range(union):
        # take half - edges_car_bike_list[i]=(0,9) dann nehmen wir das (0,9) und die (9,0) capacities
        head = edges_car_bike_list[i][0]
        tail = edges_car_bike_list[i][1]
        streetIP += (
            u_b((head, tail)) / 2 + u_c((head, tail)) + u_c((tail, head)) + u_b((tail, head)) / 2
            <= capacities[(head, tail)] * cap_factor
        )

    print("Total number of variables: " + str(streetIP.num_cols))
    print("Total number of constraints: " + str(streetIP.num_rows))

    # Objective value
    objective_bike = mip.xsum(
        f_bike(od_ind, e) * bike_time[e] * od_weighting[od_ind] for od_ind in range(len(od_flow)) for e in G.edges
    )
    objective_car = mip.xsum(
        f_car(od_ind, e) * car_time[e] * od_weighting[od_ind] for od_ind in range(len(od_flow)) for e in G.edges
    )

    if shared_lane_variables:
        # travel time on shared lanes -> weight by shared lane factor
        objective_shared = mip.xsum(
            f_shared(od_ind, e) * bike_time[e] * shared_lane_factor * od_weighting[od_ind]
            for od_ind in range(len(od_flow))
            for e in G.edges
        )
        streetIP += objective_bike + car_weight * objective_car + objective_shared
    else:
        streetIP += objective_bike + car_weight * objective_car
    return streetIP
