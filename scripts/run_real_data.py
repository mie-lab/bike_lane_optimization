import time
import os
import pandas as pd
from ebike_city_tools.optimize.utils import make_fake_od, output_to_dataframe, flow_to_df
from ebike_city_tools.optimize.linear_program import define_IP
from ebike_city_tools.utils import extend_od_matrix
from ebike_city_tools.optimize.round_simple import pareto_frontier
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
    shared_lane_factor = 2
    FLOW_CONSTANT = 1  # how much flow to send through a path (set to 0.9 since street width is oftentimes 1.8)
    OUT_PATH = "outputs"
    os.makedirs(OUT_PATH, exist_ok=True)

    # load OD
    od = pd.read_csv(os.path.join(path, "od_matrix_zolliker.csv"))
    od.rename({"osmid_origin": "s", "osmid_destination": "t"}, inplace=True, axis=1)
    od = od[od["s"] != od["t"]]

    # load nodes and edges
    nodes = gpd.read_file(os.path.join(path, "nodes_all_attributes.gpkg")).set_index("osmid")
    edges = gpd.read_file(os.path.join(path, "edges_all_attributes.gpkg"))
    edges = edges[["u", "v", "width_total_m", "maxspeed", "lanes", "length"]]
    # remove the ones with start and end at the same point
    edges = edges[edges["u"] != edges["v"]]
    # there are many 1.8 and 0.9 wide streets -> transform into 1 and 2 lane streets
    edges["width_total_m"] = edges["width_total_m"].round()  # TODO
    # fill nans of the capacity with 1
    # edges["lanes"] = edges["lanes"].fillna(1)

    # extend OD matrix because otherwise we get disconnected car graph
    od = extend_od_matrix(od, list(nodes.index))

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
    ip = define_IP(
        G,
        cap_factor=1,
        od_df=od,
        bike_flow_constant=FLOW_CONSTANT,
        car_flow_constant=FLOW_CONSTANT,
        shared_lane_factor=shared_lane_factor,
    )
    toc = time.time()
    print("Finish init", toc - tic)
    ip.optimize()
    toc2 = time.time()
    print("Finish optimization", toc2 - toc)
    print("OPT VALUE", ip.objective_value)

    # nx.write_gpickle(G, "outputs/real_G.gpickle")
    capacity_values = output_to_dataframe(ip, G)
    capacity_values.to_csv(os.path.join(OUT_PATH, "real_u_solution.csv"), index=False)
    flow_df = flow_to_df(ip, list(G.edges))
    flow_df.to_csv(os.path.join(OUT_PATH, "real_flow_solution.csv"), index=False)

    # compute the paretor frontier
    G_city = nx.MultiDiGraph(G)  # TODO
    pareto_df = pareto_frontier(
        G_city, capacity_values, shared_lane_factor=shared_lane_factor, sp_method="od", od_matrix=od
    )
    pareto_df.to_csv(os.path.join(OUT_PATH, "real_pareto_df.csv"), index=False)
