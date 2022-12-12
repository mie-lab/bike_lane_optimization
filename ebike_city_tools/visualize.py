import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import seaborn as sns
import os


def visualize_graph(G):
    pos = nx.spring_layout(G, seed=42)
    node_sizes = [3 + 10 * i for i in range(len(G))]
    M = G.number_of_edges()
    edge_colors = range(2, M + 2)
    edge_alphas = [(5 + i) / (M + 4) for i in range(M)]
    cmap = plt.cm.plasma

    nodes = nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color="indigo")
    edges = nx.draw_networkx_edges(
        G, pos, node_size=node_sizes, arrowstyle="->", arrowsize=10, edge_color=edge_colors, edge_cmap=cmap, width=2,
    )
    # set alpha value for each edge
    for i in range(M):
        edges[i].set_alpha(edge_alphas[i])

    pc = mpl.collections.PatchCollection(edges, cmap=cmap)
    pc.set_array(edge_colors)

    ax = plt.gca()
    ax.set_axis_off()
    plt.colorbar(pc, ax=ax)
    plt.show()


def plot_graph(G, directed=True, hw=0.05, weight="weight"):
    """
    Plot a graph G, 
        with edges coloured according to the edge attribute weight, 
        as arrows if directed=True,
        with arrow head width hw
    """
    weight_color = "weight" in list(list(G.edges(data=True))[0][-1].keys())

    viridis = mpl.colormaps["viridis"].resampled(8)

    node_info = [[node[0], node[1]["loc"]] for node in G.nodes(data=True)]
    node_inds = [node[0] for node in node_info]
    #     assert all(sorted(node_inds) == node_inds)
    # scatter coords
    coords = np.array([node[1] for node in node_info])
    plt.scatter(coords[:, 0], coords[:, 1])
    # plot edges
    for edge in G.edges(data=True):
        n0, n1 = coords[node_inds.index(edge[0])], coords[node_inds.index(edge[1])]
        col = viridis(edge[2]["weight"]) if weight_color else "black"
        if directed:
            plt.arrow(
                n0[0], n0[1], dx=n1[0] - n0[0], dy=n1[1] - n0[1], color=col, head_width=hw, length_includes_head=True
            )
        else:
            plt.plot([n0[0], n1[0]], [n0[1], n1[1]], c=col)
    plt.colorbar()
    plt.axis("off")
    plt.show()


def scatter_car_bike(res, metrics_for_eval, out_path=None):
    fill_functions = [lambda x: 0, lambda x: max(x) + np.std(x), lambda x: 0]
    for metric, fill_func in zip(metrics_for_eval, fill_functions):
        bike_metric, car_metric = "bike_" + metric, "car_" + metric
        fill_val = fill_func(res[bike_metric].dropna().values)
        res[bike_metric] = res[bike_metric].fillna(fill_val)
        res[car_metric] = res[car_metric].fillna(fill_val)
        plt.figure(figsize=(7, 5))
        sns.scatterplot(data=res, x="bike_" + metric, y="car_" + metric, hue="Method", s=100)
        plt.legend(title="Method")
        if out_path is None:
            plt.show()
        else:
            plt.savefig(os.path.join(out_path, metric + "_scatter.png"))