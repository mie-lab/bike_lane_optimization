# Bike lane optimization

Tools for evaluating street networks with radical redesign by splitting into bike and car lanes

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

### File description

This repository implements various baseline algorithms for splitting into car and bike lane graph. It contains the following files:

* `random_graph.py`: Functions for generating random networks. In order to make them more similar to real-world street networks, the edges are samples with a probability inversly proportional to the distance between the nodes. This ensures that streets rather connect nearby nodes. 
* Folder `optimize`: This folder contains the linear program formulation and related code
* `metrics.py`: Metrics for evaluating a given network (can be directed or undirected). So far, closeness and all-pairs shortest path distances are implemented.
* `iterative_algorithms.py`: The baseline algorithms that I implemented so far. Algorithms have varying levels of complexity, ranging from simply extracting a minimal spanning tree as the bike network to optimizing according to the betweenness centrality.
* `rl_env.py`: At some point I wanted to train a reinforcemnt learning agent to improve bike networks. This is only the environment that could be used for that (the RL agent is not implemented yet).

The script `scripts/test.py` can be executed to test the optimization algorithm on random data.

For example, run 
```
python scripts/test.py
```

Most scripts will save the results in the `outputs` folder.

### Explanatory notes

There are two types of graph structures used throughout the code:
* lane graphs: nx.MultiDiGraph, usually denoted as G_lane, one directed edge per lane
* street graphs: nx.DiGraph, usually denoted as G_street, two reciprocal edges per street (this is the input to the optimization algorithm)