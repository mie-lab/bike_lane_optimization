import time
import os
import pandas as pd
from ebike_city_tools.random_graph import city_graph, lane_to_street_graph
from ebike_city_tools.optimize.utils import make_fake_od, output_to_dataframe, flow_to_df
from ebike_city_tools.optimize.linear_program import initialize_IP
import numpy as np
import geopandas as gpd
import networkx as nx


def table_to_graph(
    edge_table, node_table=None, edge_attributes={"width_total_m": "capacity"}, node_attributes={"geometry": "location"}
):
    """
    edge_table: pd.DataFrame with columns u, v, and the required edge attributes
    node_table (Optional): table with the node id as the index column and the edge id as the
    edge_attributes: Dictionary of the form {columns-name-in-table : desired-attribute_name-in-graph}
    node_attributes: Dictionary of the form {columns-name-in-table : desired-attribute_name-in-graph}
    """
    # init graph
    G = nx.DiGraph()

    # add edge list
    edge_list = []
    for row_ind, edge_row in edge_table.iterrows():
        # extract the edge attributes
        edge_attr = {attr_name: edge_row[col_name] for col_name, attr_name in edge_attributes.items()}
        # add edge with attributes to the list
        edge_list.append([edge_row["u"], edge_row["v"], edge_attr])
        edge_attr["gradient"] = -edge_attr["gradient"]
        edge_list.append([edge_row["v"], edge_row["u"], edge_attr])
    G.add_edges_from(edge_list)

    # set node attributes
    node_attrs = {}
    for row_ind, node_row in node_table.iterrows():
        node_attrs[row_ind] = {attr_name: node_row[col_name] for col_name, attr_name in node_attributes.items()}
    nx.set_node_attributes(G, node_attrs)
    return G


if __name__ == "__main__":
    path = "../street_network_data/ressources_aurelien"

    # load OD
    od = pd.read_csv(os.path.join(path, "od_matrix_zolliker.csv"))
    od.rename({"osmid_origin": "s", "osmid_destination": "t"}, inplace=True, axis=1)
    od = od[od["s"] != od["t"]]

    # load nodes and edges
    nodes = gpd.read_file(os.path.join(path, "nodes_all_attributes.gpkg")).set_index("osmid")
    edges = gpd.read_file(os.path.join(path, "edges_all_attributes.gpkg"))
    edges = edges[["u", "v", "width_total_m", "maxspeed", "lanes", "length"]]
    # fill nans of the capacity with 1
    # edges["lanes"] = edges["lanes"].fillna(1)

    # # making a subgraph only disconnects the graoh
    # nodes = nodes.sample(200)
    # edges = edges[edges["u"].isin(nodes.index)]
    # edges = edges[edges["v"].isin(nodes.index)]
    # od = od[od["s"].isin(nodes.index)]
    # od = od[od["t"].isin(nodes.index)]

    # compute gradient
    gradient = []
    for i in range(len(edges)):
        gradient.append(
            100 * (nodes["elevation"][edges.iloc[i, 1]] - nodes["elevation"][edges.iloc[i, 0]]) / edges.iloc[i, 5]
        )
    edges["gradient"] = gradient

    # construct graph
    G = table_to_graph(edges, nodes, {"width_total_m": "capacity", "length": "distance", "gradient": "gradient"})

    assert nx.is_strongly_connected(G), "G not connected"

    tic = time.time()
    ip = initialize_IP(G, cap_factor=1, od_df=od, bike_flow_constant=0.9, car_flow_constant=0.9)
    toc = time.time()
    print("Finish init", toc - tic)
    ip.optimize()
    toc2 = time.time()
    print("Finish optimization", toc2 - toc)
    print("OPT VALUE", ip.objective_value)

    # nx.write_gpickle(G, "outputs/real_G.gpickle")
    dataframe_edge_cap = output_to_dataframe(ip, G)
    dataframe_edge_cap.to_csv("outputs/real_u_solution.csv", index=False)
    flow_df = flow_to_df(ip, list(G.edges))
    flow_df.to_csv("outputs/real_flow_solution.csv", index=False)
