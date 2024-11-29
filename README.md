# Bike network planning in limited urban space

This code accompanies our [paper](https://arxiv.org/abs/2405.01770) **Bike Network Planning in Limited Urban Space**, accepted to *Transportation Research Part B*. 

This project presents an optimization framework for planning bike lane networks in urban areas, aiming to minimize the impact on other transport modes. We model the problem as a linear program, where the objective function balances bike and car travel times through a weighted sum. Travel times are calculated using network distances and speed limits, with the assumption that cyclists can use all roads. However, roads without dedicated bike lanes have higher *perceived* travel times for cyclists.

The trade-off between bike and car travel times can be visualized with Pareto frontiers. As shown in the figure below, our algorithm improves the pareto-optimality over the tested baselines. It allows to radically rebuild the street network, repurposing a significant proportion of lanes as bike lanes while maintaining the connectivity of the car network. 

<img src="assets/overview.png" alt="Pareto frontier and corresponding street networks" style="width:100%;">

This project is part of the [E-Bike City Project](https://ebikecity.baug.ethz.ch/en/), a multi-disciplinary research project at ETH Zürich exploring the effects of an urban future giving absolute priority to cycling, micromobility and public transport. 

### Installation

The code can be installed via pip in editable mode in a virtual environment with the following commands:

```
git clone https://github.com/mie-lab/bike_lane_optimization
cd  bike_lane_optimization
python -m venv env
source env/bin/activate
pip install -e .
````

This installs the package called `ebike_city_tools` in your virtual environment, together with all dependencies. 
The functions in this package can then be imported from any folder, e.g. `from ebike_city_tools.metrics import *`

We also provide a Dockerfile that allows deploying this code as a Python Flask app. We are working on a [web app](https://ebikecity.webapp.ethz.ch) that provides an easy-to-use interface to our algorithms.

### Simple testing on random data

The script `scripts/test.py` can be executed to test the optimization algorithm on random data.

For example, run 
```
python scripts/test.py
```

Most scripts will save the results to the `outputs` folder by default.

### Running the algorithm on real data

Three instances of street networks in the zity of Zurich can be downloaded [here](https://polybox.ethz.ch/index.php/s/YaoJkHRofZKUiTG).

To preprocess the data, the [SNMan](https://github.com/lukasballo/snman) package is required. Installation instructions can be found in their repo.

After downloading the data and installing SNMan, the algorithm can be executed by running `run_real_data.py` with the path to the data specified via the `-d` flag (per default `../street_network_data/zollikerberg` -> set to the path where you downloaded the data).

Usage:
```
python scripts/run_real_data.py [-h] [-d DATA_PATH] [-i INSTANCE] [-o OUT_PATH] [-k OPTIMIZE_EVERY_K] [-c CAR_WEIGHT] [-v VALID_EDGES_K] [-p PENALTY_SHARED] [-s SP_METHOD] [--save_graph] [-a ALGORITHM]

optional arguments:
 -h, --help            show this help message and exit
  -d DATA_PATH, --data_path DATA_PATH
  -i INSTANCE, --instance INSTANCE
  -o OUT_PATH, --out_path OUT_PATH
  -k OPTIMIZE_EVERY_K, --optimize_every_k OPTIMIZE_EVERY_K
                        how often to re-optimize
  -c CAR_WEIGHT, --car_weight CAR_WEIGHT
                        weighting of cars in objective function
  -v VALID_EDGES_K, --valid_edges_k VALID_EDGES_K
                        if subsampling edges, the number of nodes around SP
  -p PENALTY_SHARED, --penalty_shared PENALTY_SHARED
                        penalty factor for driving on a car lane by bike
  -s SP_METHOD, --sp_method SP_METHOD
                        Compute the shortest path either 'all_pairs' or 'od'
  --save_graph          if true, only creating one graph and saving it
  -a ALGORITHM, --algorithm ALGORITHM
                        One of optimize, betweenness_topdown, betweenness_cartime, betweenness_biketime
```

For example, if you downloaded the data for Affoltern and put it into a folder `data/affoltern`, you can execute it with
```
python scripts/run_real_data.py -d data -i affoltern -o outputs
```

It will use the default parameters for `optimize_every_k`, `car_weight`, `valid_edges_k`, `penalty_shared`, `sp_method` (see paper for parameter definitions), and save the pareto frontier into the folder `outputs`.

### Explanatory notes

There are two types of graph structures used throughout the code:
* lane graphs: nx.MultiDiGraph, usually denoted as G_lane, one directed edge per lane
* street graphs: nx.DiGraph, usually denoted as G_street, two reciprocal edges per street (essentially an undirected graph)

We start and end with a lane graph, reflecting the current situation of the network and the rebuilt version respectivaly. Our algorithm is, however, applied on the street graph, allowing the algorithm to allocate the optimal number of lanes in either directions based on the street capacity.

#### File description

`scripts`: Folder with scripts used to produce the results in our paper

`ebike_city_tools`: Package with main code, including:
* Folder `optimize`: This folder contains the linear program formulation (see [here](ebike_city_tools/optimize/linear_program.py)) and related code
* `synthetic.py`: Functions for generating random networks for testing. In order to make them more similar to real-world street networks, the edges are samples with a probability inversly proportional to the distance between the nodes. This ensures that streets rather connect nearby nodes. 
* `metrics.py`: Metrics for evaluating a given network (can be directed or undirected). Mainly based on network-based travel times (and perceived bike travel time)
* `iterative_algorithms.py`: The baseline algorithms, all based on the betweenness centrality heuristic.

`app.py`: Code to deploy a Python Flask app

### References

**Please consider citing our paper if you  build up on this work:**

Wiedemann, N., Nöbel, C., Martin, H., Ballo, L., & Raubal, M. (2024). Bike network planning in limited urban space. arXiv preprint arXiv:2405.01770.


```bib
@article{wiedemann2024bike,
  title={Bike network planning in limited urban space},
  author={Wiedemann, Nina and N{\"o}bel, Christian and Martin, Henry and Ballo, Lukas and Raubal, Martin},
  journal={arXiv preprint arXiv:2405.01770},
  year={2024}
}
```