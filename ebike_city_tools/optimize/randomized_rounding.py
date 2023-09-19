import networkx as nx
import pandas as pd
import numpy as np
import random

from ebike_city_tools.utils import add_bike_and_car_time
from ebike_city_tools.metrics import od_sp

from ebike_city_tools.optimize.rounding_utils import build_bike_network_from_df, build_car_network_from_df, edge_to_source_target, result_to_streets, repeat_and_edgekey

def apply_randomized_rounding_to_graph(result_df):
    # transform into dataframe of undirected edges
    street_df = result_to_streets(result_df.copy()).reset_index()

    print(street_df.head)
    # Apply the randomized rounding to the edges
    rounded_df = street_df.apply(random_round_one_edge, axis = 1)

    print(rounded_df.head)

    g_car = build_car_network_from_df(rounded_df.copy())
    g_bike = build_bike_network_from_df(rounded_df)

    return g_car, g_bike

def randomized_rounding(result_df):
    # Initial car graph is one with all the car capacity values rounded up
    for i in range(10) :
        random.seed(i)
        car_G, bike_G = apply_randomized_rounding_to_graph(result_df.copy())
        if nx.is_strongly_connected(car_G):
            break
        print("DID NOT YIELD STRONGLY CONNECTED GRAPH")

    assert nx.is_strongly_connected(car_G)

    print("Start graph edges", bike_G.number_of_edges(), car_G.number_of_edges())

    return bike_G, car_G

def random_round_one_edge(data_frame_row):
    #Auxiliary function, that rounds the capacities for a single edge.
    #Rounding is done randomized, with the fractional part being the probability of rounding up.

    #Introduce shortcuts
    u_c = data_frame_row["u_c(e)"]
    u_c_r = data_frame_row["u_c(e)_reversed"]
    u_b = data_frame_row["u_b(e)"]

    #Calculate the rounding probabilities
    p = u_c - np.floor(u_c)
    q = u_c_r - np.floor(u_c_r)

    #Do the rounding
    random_number = random.uniform(0,1)
    
    if random_number <= p :
        u_c = u_c + 1
    elif random_number <= p+q:
        u_c_r = u_c_r + 1
    else:
        u_b = u_b + 1

    return pd.Series([data_frame_row["Edge"], u_c, u_c_r, u_b, data_frame_row["capacity"]], index=['Edge', 'u_c(e)', 'u_c(e)_reversed', 'u_b(e)', 'capacity'])

