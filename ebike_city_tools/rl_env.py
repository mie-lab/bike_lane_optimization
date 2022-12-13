from ebike_city_tools.metrics import *


class StreetNetworkEnv:
    def __init__(self, orig_graph, initial_setting=None):
        self.orig_graph = orig_graph.copy()
        self.initial_setting = initial_setting
        # action space: car->bike, car_rev->bike, bike->car, car_rev->car, bike->car_rev, car->car_rev
        self.n_action_types = 6
        self.opposite_dict = {0: 2, 1: 4, 2: 0, 4: 1, 3: 5, 5: 3}
        self.avail_action_by_current = {"bike": [2, 4], "car": [0, 5], "car_reversed": [1, 3]}
        self.n_actions = self.orig_graph.number_of_edges() * self.n_action_types
        self.orig_edges = [(i, e) for i, e in enumerate(list(orig_graph.edges(keys=True)))]
        # reset
        self.reset()

    def derive_initial_setting(self, init_car_graph):
        init_car_edges = list(init_car_graph.edges(keys=True))
        self.initial_setting = {(i, e): "car" if e in init_car_edges else "bike" for (i, e) in self.orig_edges}

    def reset(self):
        self.current_assignment = self.initial_setting
        if self.current_assignment is None:
            # random assignment
            self.current_assignment = {
                ind_and_e: "bike" if np.random.rand() < 0.5 else "car" for ind_and_e in self.orig_edges
            }
        edge_list_bike = [(e[0], e[1], i, {}) for i, e in self.orig_edges if self.current_assignment[(i, e)] == "bike"]
        edge_list_car = [(e[0], e[1], i, {}) for i, e in self.orig_edges if self.current_assignment[(i, e)] == "car"]
        # car graph: directed graph without the bike lanes
        self.car_graph = self.orig_graph.copy()
        self.car_graph.remove_edges_from(list(self.car_graph.edges()))
        self.car_graph.add_edges_from(edge_list_car)

        # bike graph: undirected graph with the bike lanes
        node_attributes = nx.get_node_attributes(self.orig_graph, name="loc")
        self.bike_graph = nx.MultiGraph()
        self.bike_graph.add_edges_from(edge_list_bike)
        #         print(edge_list_bike)
        #         print(self.bike_graph.edges())
        #         print("--------")
        nx.set_node_attributes(self.bike_graph, node_attributes, name="loc")

        # init available actions
        self.actions_avail = np.array(
            [self.avail_action_by_current[self.current_assignment[edge]] for edge in self.orig_edges]
        )

        # init reward
        self.last_reward = self.compute_reward()

    def get_available_actions(self):
        """Return a list of all available actions"""
        n_edges, n_avail_per_edge = self.actions_avail.shape
        avail_act = np.zeros(n_edges * n_avail_per_edge).astype(int)
        for i in range(n_avail_per_edge):  # number available per edge
            avail_act[i * n_edges : (i + 1) * n_edges] = (
                np.arange(len(self.orig_edges)) * self.n_action_types + self.actions_avail[:, i]
            )
        return avail_act

    def step(self, action):
        edge_of_action = self.orig_edges[action // self.n_action_types]
        edge_key, (edge_source, edge_target, _) = edge_of_action
        action_type = action % self.n_action_types
        #         print(edge_of_action, action_type)

        changed = False  # check if we actually changed something or stayed the same
        if action_type == 0 or action_type == 5:
            # currenlty car, switch
            if self.current_assignment[edge_of_action] == "car":
                self.car_graph.remove_edge(edge_source, edge_target, key=edge_key)
                changed = True
                if action_type == 0:  # car -> bike
                    self.bike_graph.add_edge(edge_source, edge_target, key=edge_key)
                    self.current_assignment[edge_of_action] = "bike"
                elif action_type == 5:  # car -> car_reversed
                    self.car_graph.add_edge(edge_target, edge_source, key=edge_key)
                    self.current_assignment[edge_of_action] = "car_reversed"
        if action_type == 2 or action_type == 4:
            # currenlty bike, switch
            if self.current_assignment[edge_of_action] == "bike":
                # remove edge from bike graph
                self.bike_graph.remove_edge(edge_source, edge_target, key=edge_key)
                changed = True
                if action_type == 2:  # bike -> car
                    self.car_graph.add_edge(edge_source, edge_target, key=edge_key)
                    self.current_assignment[edge_of_action] = "car"
                elif action_type == 4:  # bike -> car_reversed
                    self.car_graph.add_edge(edge_target, edge_source, key=edge_key)
                    self.current_assignment[edge_of_action] = "car_reversed"
        if action_type == 1 or action_type == 3:
            # currenlty car_reversed, switch
            if self.current_assignment[edge_of_action] == "car_reversed":
                # remove edge from car graph
                self.car_graph.remove_edge(edge_target, edge_source, key=edge_key)
                changed = True
                if action_type == 1:  # car_reversed -> bike
                    self.bike_graph.add_edge(edge_source, edge_target, key=edge_key)
                    self.current_assignment[edge_of_action] = "bike"
                elif action_type == 3:  # car_reversed -> car
                    self.car_graph.add_edge(edge_source, edge_target, key=edge_key)
                    self.current_assignment[edge_of_action] = "car"

        if not changed:
            reward = self.last_reward
        else:
            # update action availability
            self.actions_avail[edge_key] = self.avail_action_by_current[self.current_assignment[edge_of_action]]
            # get reward
            reward = self.compute_reward()
            self.last_reward = reward
        #         if changed:
        #             print("Bike", self.bike_graph.edges(keys=True))
        #             print("Car", self.car_graph.edges())
        #         print()
        return reward

    def revert_action(self, action):
        action_type = action % self.n_action_types
        changed_edge = action // self.n_action_types
        opposite_action = self.opposite_dict[action_type]
        return self.step(changed_edge * self.n_action_types + opposite_action)

    def compute_reward(self):
        # TODO: more precise rewards
        # ["sp_reachability", "sp_length", "closeness"]
        bike_closeness = closeness(self.bike_graph)
        car_closeness = closeness(self.car_graph)
        return bike_closeness + car_closeness


if __name__ == "__main__":
    from random_graph import base_graph_doppelspur
    import matplotlib.pyplot as plt

    base_graph = base_graph_doppelspur(20)
    env = StreetNetworkEnv(base_graph)
    env.reset()
    prev_reward = 0
    rewards = []
    for i in range(1000):

        act_avail = env.get_available_actions()
        #     action = np.random.randint(env.n_actions)
        action = np.random.choice(act_avail)

        rew = env.step(action)
        if rew < prev_reward:
            rew = env.revert_action(action)
            assert rew >= prev_reward  # TODO: understand why this fails sometimes

        prev_reward = rew
        rewards.append(rew)
    plt.plot(rewards)
    plt.show()
