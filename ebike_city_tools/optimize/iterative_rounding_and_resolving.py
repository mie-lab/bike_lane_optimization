import math
import pandas as pd

from ebike_city_tools.utils import output_to_dataframe
from ebike_city_tools.optimize.round_simple import pareto_frontier
from ebike_city_tools.optimize.rounding_utils import result_to_streets, undirected_to_directed 

def rounding_error(fractional_value):
    next_int = round(fractional_value)
    return abs(next_int - fractional_value)

def rounding_error_of_row(row):
    return rounding_error(row['u_c(e)'])

def round_row(row):
    # print(row)
    if row['rounding_error'] + row['rounding_error_reversed'] < 1 :
        row['u_c(e)'] = round(row['u_c(e)'])
        row['u_c(e)_reversed'] = round(row['u_c(e)_reversed'])
        row['u_b(e)'] = float(row['capacity'] - row['u_c(e)'] - row['u_c(e)_reversed'])/2
    else:
        row['u_c(e)'] = math.ceil(row['u_c(e)'])
        row['u_c(e)_reversed'] = math.floor(row['u_c(e)_reversed'])
    return row

def iterative_rounding(optimizer, G_lane, shared_lane_factor,
                       fraction_of_edges_rounded = 0.5,number_iterations = 5):

    pareto_fronts = []
    directed_fixed_df = pd.DataFrame()

    for _ in range(number_iterations):
        optimizer.init_lp_with_fixed_edges(directed_fixed_df)
        optimizer.optimize()
        print(optimizer.lp.objective_value)
        if optimizer.lp.objective_value is None :
            print("LP infeasible after rounding")
            break
        capacity_values = output_to_dataframe(optimizer.lp, optimizer.graph, fixed_edges = directed_fixed_df)

        optimizer.fixed_edges = directed_fixed_df  
        pareto_fronts.append(pareto_frontier(G_lane, capacity_values, shared_lane_factor))
        capacity_values['rounding_error'] = capacity_values.apply(rounding_error_of_row, axis = 1)

        combined_df = result_to_streets(capacity_values)
        combined_df['total_rounding_error'] = combined_df['rounding_error'] + combined_df['rounding_error_reversed']

        fractional_df = combined_df[combined_df['total_rounding_error'] > 0]
        integral_df = combined_df[combined_df['total_rounding_error'] == 0]

        if fractional_df.shape[0] == 0:
            print("No more entries to round, stopping iterative_rounding")
            break
        number_to_round = min(int(fractional_df.shape[0]*fraction_of_edges_rounded),fractional_df.shape[0]-1)

        to_round_df = fractional_df.sort_values(by=['total_rounding_error']).take(range(0,number_to_round))
        rounded_df = to_round_df.apply(round_row, axis = 1)

        fixed_df = pd.concat([rounded_df, integral_df])
        print("Number of fixed Edges: ", fixed_df.shape[0])

        directed_fixed_df = undirected_to_directed(fixed_df).reset_index()
        


    return pareto_fronts 
