import math
import numpy as np
import pandas as pd
from pandas.core.construction import com

from ebike_city_tools.optimize.utils import output_to_dataframe, flow_to_df
from ebike_city_tools.optimize.round_simple import rounding_and_splitting
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

def iterative_rounding(result_df, optimizer):
    FRACTION_TO_ROUND = 0.25
    NUMBER_ITERATIONS = 2

    capacity_values = result_df
    for _ in range(NUMBER_ITERATIONS):
        capacity_values['rounding_error'] = capacity_values.apply(rounding_error_of_row, axis = 1)

        combined_df = result_to_streets(capacity_values)
        combined_df['total_rounding_error'] = combined_df['rounding_error'] + combined_df['rounding_error_reversed']

        fractional_df = combined_df[combined_df['total_rounding_error'] > 0.001]
        integral_df = combined_df[combined_df['total_rounding_error'] <= 0.001]

        number_to_round = int(fractional_df.shape[1]*FRACTION_TO_ROUND)

        to_round_df = fractional_df.sort_values(by=['total_rounding_error']).take(range(0,number_to_round))
        print("To round")
        print(to_round_df)
        rounded_df = to_round_df.apply(round_row, axis = 1)
        print("Rounded:")
        print(rounded_df)

        fixed_df = pd.concat([rounded_df, integral_df])
        print(fixed_df)

        directed_fixed_df = undirected_to_directed(fixed_df).reset_index()
        print(directed_fixed_df)

        optimizer.init_lp_with_fixed_edges(directed_fixed_df )
        optimizer.optimize()
        print(optimizer.lp.objective_value)
        capacity_values = output_to_dataframe(optimizer.lp, optimizer.graph, fixed_edges = directed_fixed_df)
        print(capacity_values.take(range(10)))
        # print(result_df.head)

        optimizer.fixed_edges = directed_fixed_df  

    return rounding_and_splitting(capacity_values)
