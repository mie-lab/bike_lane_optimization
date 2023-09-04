import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
import os

plt.rcParams.update({"font.size": 15})


def visualize_graph(G):
    pos = nx.spring_layout(G, seed=42)
    node_sizes = [3 + 10 * i for i in range(len(G))]
    M = G.number_of_edges()
    edge_colors = range(2, M + 2)
    edge_alphas = [(5 + i) / (M + 4) for i in range(M)]
    cmap = plt.cm.plasma

    nodes = nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color="indigo")
    edges = nx.draw_networkx_edges(
        G,
        pos,
        node_size=node_sizes,
        arrowstyle="->",
        arrowsize=10,
        edge_color=edge_colors,
        edge_cmap=cmap,
        width=2,
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


def scatter_car_bike(res, metrics_for_eval, out_path="figures"):
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


def pareto_plot_sp(res, out_path="figures"):
    """Plot with dotted lines for the extreme cases and scatter points otherwise"""

    car_optim = res.set_index("Method").loc["original", "car_sp_length"]
    bike_optim = res.set_index("Method").loc["full_random", "bike_sp_length"]

    res = res[~res["Method"].isin(["original", "full_random"])]

    method_mapping = {
        "full_random": "One bi-directional bike lane per street",
        "spanning_random": "Minimal spanning tree forms bike network",
        "betweenness": "Bike lanes found by minimum betweenness centrality [1]",
        "optim_betweenness": "Randomly swap edges to improve [1]",
        "original": "No bike lanes (original car network)",
    }
    res["Method"] = res["Method"].map(method_mapping)
    # sort
    res = (
        res.set_index(["Method"])
        .loc[
            [
                "Minimal spanning tree forms bike network",
                "Bike lanes found by minimum betweenness centrality [1]",
                "Randomly swap edges to improve [1]",
            ]
        ]
        .reset_index()
    )

    metric = "sp_length"
    xlim = 6.4
    ylim = 4.5
    plt.figure(figsize=(14, 5))
    sns.scatterplot(data=res, x="bike_" + metric, y="car_" + metric, hue="Method", s=100, palette="Set2")
    plt.plot([0, xlim], [car_optim, car_optim], label=method_mapping["original"], linestyle="--", color="red")
    plt.plot([bike_optim, bike_optim], [0, ylim], label=method_mapping["full_random"], linestyle="--", color="green")
    plt.legend(title="Method", bbox_to_anchor=(1, 0.8))
    plt.ylabel("Avg. shortest path length (CAR)")
    plt.xlabel("Avg. shortest path length (BIKE)")
    plt.xlim(0, xlim)
    plt.ylim(0, ylim)
    plt.tight_layout()
    plt.savefig(os.path.join(out_path, "pareto_sp.jpg"))
    plt.show()


def visualize_runtime_dependency(path="outputs/runtime.csv", out_path="figures"):
    """Plot the runtime for initialization and optimization"""
    runtime = pd.read_csv(path)
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.scatter(runtime["od_size"], runtime["time init"], c=runtime["edges"])
    plt.xlabel("Number of s-t-paths (size of OD matrix)")
    plt.ylabel("LP initialization runtime [s]")
    plt.colorbar(label="Number of edges")
    plt.subplot(1, 2, 2)
    plt.scatter(runtime["nodes"], runtime["time optimize"], c=runtime["edges"])
    plt.xlabel("Number of s-t-paths (size of OD matrix)")
    plt.ylabel("Optimization runtime [s]")
    plt.colorbar(label="Number of edges")
    plt.tight_layout()
    plt.savefig(os.path.join(out_path, "runtime_analysis.png"))


def visualize_od_dependency(path="outputs/od_dependency.csv", out_path="figures"):
    """Function to visualize the OD dependency"""
    od_dependency = pd.read_csv(path)
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.scatter(od_dependency["od reduction"], od_dependency["runtime_init"], c="blue", label="init")
    plt.scatter(od_dependency["od reduction"], od_dependency["runtime_optim"], c="orange", label="optim")
    plt.ylabel("runtime")
    plt.xlabel("OD reduction factor")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.scatter(od_dependency["od reduction"], od_dependency["bike sp lengths"], c="blue", label="bike")
    plt.scatter(od_dependency["od reduction"], od_dependency["car sp lengths"], c="orange", label="car")
    plt.ylabel("SP lengths")
    plt.xlabel("OD reduction factor")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_path, "od_dependency_analysis.png"))

    od_dependency.groupby("od reduction").agg(
        {
            "car sp reachable": "mean",
            "bike sp reachable": "mean",
            "bike sp lengths": ["mean", "std"],
            "car sp lengths": ["mean", "std"],
        }
    ).to_csv(os.path.join(out_path, "od_nonreachable.csv"))


def compare_pareto(in_path="outputs", out_path="figures"):
    for od in [True, False]:
        ending = "_od" if od else ""
        pareto_ours = pd.read_csv(os.path.join(in_path, f"real_pareto_df{ending}.csv"))
        pareto_between = pd.read_csv(os.path.join(in_path, f"real_pareto_betweenness{ending}.csv"))
        plt.figure(figsize=(6, 5))
        plt.scatter(pareto_ours["bike_time"], pareto_ours["car_time"], label="Ours")
        plt.scatter(pareto_between["bike_time"], pareto_between["car_time"], label="Betweenness")
        plt.xlabel("Bike travel time")
        plt.ylabel("Car travel time")
        plt.xlim(35, 75)
        plt.ylim(15, 55)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_path, f"comparison{ending}.pdf"))
        plt.show()


def compare_int_lin(in_path="outputs/integer_vs_linear.csv", out_path="figures"):
    int_lin = pd.read_csv(in_path)
    res = []
    # compute distance of linear to pareto frontier
    for group, group_df in int_lin.groupby(["iter", "nodes", "edges", "od_size"]):
        # plot pareto frontier
        group_df_linear = group_df[~pd.isna(group_df["bike_edges_added"])]
        # plot integer as scatter
        integer_bike = group_df.loc[pd.isna(group_df["bike_edges_added"]), "bike_time"]
        integer_car = group_df.loc[pd.isna(group_df["bike_edges_added"]), "car_time"]

        # sometimes, we only have the result for int or the result for lin
        if len(integer_car) == 0 or len(group_df_linear) == 0:
            continue

        # plot exactly one group
        if group == (1, 40, 146, 145):
            plt.figure(figsize=(6, 5))
            plt.scatter(group_df_linear["car_time"], group_df_linear["bike_time"], label="linear")
            plt.scatter([integer_car], [integer_bike], label="integer")
            plt.legend()
            plt.xlabel("car travel time")
            plt.ylabel("bike travel time")
            plt.tight_layout()
            plt.savefig(os.path.join(out_path, "pareto_example.pdf"))
            # plt.show()

        closest_val = abs(group_df_linear["car_time"] - integer_car.values[0]).idxmin()
        int_closest = [integer_bike.values[0], integer_car.values[0]]
        lin_closest = group_df_linear.loc[closest_val][["bike_time", "car_time"]].values
        if int_closest[0] > 1.01 * lin_closest[0] or int_closest[1] > 1.01 * lin_closest[1]:
            # plt.scatter(group_df_linear["car_time"], group_df_linear["bike_time"], label="linear")
            # plt.scatter([integer_car], [integer_bike], label="integer")
            # plt.legend()
            # plt.show()
            print("skipping because substantial difference betweeen linear and integer")
            continue
        else:
            int_better = np.array(lin_closest) - np.array(int_closest)
            dist = np.linalg.norm(int_better)
            res.append([dist] + list(int_better) + list(int_better / np.array(lin_closest)))
    # sumarize in dataframe
    res = pd.DataFrame(
        res,
        columns=[
            "Vector distance",
            "Difference bike time",
            "Difference car time",
            "Relative difference bike time",
            "Relative difference car time",
        ],
    ).round(3)
    res.index.name = "trial"

    # plot distribution
    plt.figure(figsize=(6, 5))
    ax = plt.subplot(111)
    sns.kdeplot(res["Relative difference bike time"], ax=ax)
    ax.set_ylabel("Density (bike time)")
    ax2 = ax.twinx()
    ax2.set_ylabel("Density (car time)")
    sns.kdeplot(res["Relative difference car time"], ax=ax2, c="orange", label="Relative difference car time")
    ax2.set_xlabel("Relative difference")
    ax.set_xlabel("Relative difference")
    plt.tight_layout()
    plt.savefig(os.path.join(out_path, "lin_int_diff.pdf"))
    plt.show()


def integer_linear_examples(in_path="outputs/integer_vs_linear.csv", out_path="figures/pareto_examples"):
    int_vs_lin = pd.read_csv(in_path)
    os.makedirs(out_path, exist_ok=True)
    i = 0
    for (nodes, edges, od_size), part_df in int_vs_lin.groupby(["nodes", "edges", "od_size"]):
        integer = part_df[part_df["name"] == "integer"]
        linear = part_df[part_df["name"] == "linear"]
        plt.scatter(linear["bike_time"] * 60, linear["car_time"] * 60, c=linear["car_weight"].values, marker=".", label="linear") #  c=linear["bike_edges"].values
        plt.scatter(integer["bike_time"] * 60, integer["car_time"] * 60, c=integer["car_weight"].values, marker="x", label="integer") # c=integer["car_weight"].values,
        plt.xlabel("Avg. bike travel time [min]")
        plt.ylabel("Avg. car travel time [min]")
        plt.legend(title="IP or LP formulation")
        plt.colorbar(label="weighting of car time in objective")
        plt.tight_layout()
        plt.savefig(os.path.join(out_path, f"example_{i}.pdf"))
        plt.show()
        i += 1


if __name__ == "__main__":
    compare_int_lin()
    visualize_runtime_dependency()
    visualize_od_dependency()
    compare_pareto()
