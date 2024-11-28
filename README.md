# Bike network planning in limited urban space

This code accompanies our paper `Bike Network Planning in Limited Urban Space`, accepted to Transportation Research Part B. 

We developed an optimization approach for placing bike lanes in cities, with the goal of minimizing the impact on other transport modes. We propose a linear program where the objective function is a weighted sum of the bike and car travel times. Travel times are measured as network distances combined with the speed limit. We assume that cyclists can take all roads, but the bike travel time along a street is perceived to be higher if there is no dedicated bike lane. 

The trade-off between bike and car travel times can be visualized with Pareto frontiers. As shown in the figure below, our algorithm improves the pareto-optimality over the tested baselines. It allows to radically rebuild the street network, repurposing a large fraction of lanes as bike lanes without interrupting the connectivity of the car network. 


<img src="assets/overview.png" alt="Pareto frontier and corresponding street networks" style="width:100%;">


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

We also provide a Dockerfile that allows deploying this code as a Python Flask app.

### File description

This repository implements various baseline algorithms for splitting into car and bike lane graph. It contains the following files:

* `synthetic.py`: Functions for generating random networks. In order to make them more similar to real-world street networks, the edges are samples with a probability inversly proportional to the distance between the nodes. This ensures that streets rather connect nearby nodes. 
* Folder `optimize`: This folder contains the linear program formulation and related code
* `metrics.py`: Metrics for evaluating a given network (can be directed or undirected). So far, closeness and all-pairs shortest path distances are implemented.
* `iterative_algorithms.py`: The baseline algorithms that I implemented so far. Algorithms have varying levels of complexity, ranging from simply extracting a minimal spanning tree as the bike network to optimizing according to the betweenness centrality.
* `rl_env.py`: At some point I wanted to train a reinforcemnt learning agent to improve bike networks. This is only the environment that could be used for that (the RL agent is not implemented yet).

### Simple testing on random data

The script `scripts/test.py` can be executed to test the optimization algorithm on random data.

For example, run 
```
python scripts/test.py
```

Most scripts will save the results in the `outputs` folder.

### Running the algorithm on real data

Three instances of street networks in the zity of Zurich can be downloaded [here](https://polybox.ethz.ch/index.php/s/YaoJkHRofZKUiTG).

To preprocess the data, the [SNMan](https://github.com/lukasballo/snman) package is required. Installation instructions can be found in the README file.

After downloading the data and installing SNMan, the algorithm can be executed by running `run_real_data.py` with the path to the data specified via the `-d` flag (per default `../street_network_data/zollikerberg` -> set to the path where you downloaded the data).

Usage:
```
python scripts/run_real_data.py [-h] [-d DATA_PATH] [-o OUT_PATH] [-b] [-p PENALTY_SHARED] [-s SP_METHOD]

optional arguments:
  -h, --help            show this help message and exit
  -d DATA_PATH, --data_path DATA_PATH
  -o OUT_PATH, --out_path OUT_PATH
  -b, --run_betweenness
  -p PENALTY_SHARED, --penalty_shared PENALTY_SHARED
                        penalty factor for driving on a car lane by bike
  -s SP_METHOD, --sp_method SP_METHOD
                        Compute the shortest path either 'all_pairs' or 'od'
```

### Explanatory notes

There are two types of graph structures used throughout the code:
* lane graphs: nx.MultiDiGraph, usually denoted as G_lane, one directed edge per lane
* street graphs: nx.DiGraph, usually denoted as G_street, two reciprocal edges per street (this is the input to the optimization algorithm)


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