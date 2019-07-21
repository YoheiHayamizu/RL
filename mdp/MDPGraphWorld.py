from RL.mdp.MDPBasis import MDPBasisClass
from RL.mdp.MDPState import MDPStateClass
from RL.mdp.GraphWorldConstants import *
import random
import networkx as nx


class MDPGraphWorld(MDPBasisClass):
    def __init__(self, node_num=15, init_node=0,
                 goal_nodes=goal_nodes_tuple,
                 has_door_nodes=has_door_nodes_tuple,
                 door_open_nodes=door_open_nodes_dict,
                 door_id=door_id_dict,
                 is_goal_terminal=True, success_rate=success_rate_dict1, step_cost=1.0, name="Graphworld"
                 ):
        self.actions = ACTIONS
        self.node_num = node_num
        self.is_goal_terminal = is_goal_terminal
        self.success_rate = success_rate
        self.goal_nodes = goal_nodes
        self.has_door_nodes = has_door_nodes
        self.door_open_nodes = door_open_nodes
        self.door_id = door_id
        self.step_cost = step_cost
        self.name = name
        self.nodes = self.set_nodes()
        self.init_state = self.nodes[init_node]
        self.cur_state = self.nodes[init_node]
        self.G = self.set_graph()
        super().__init__(self.init_state, self.actions, self._transition_func, self._reward_func,
                         self.step_cost)

    def __str__(self):
        return self.name + "_n-" + str(self.node_num)

    def __repr__(self):
        return self.__str__()

    # Accessors

    def get_params(self):
        get_params = super().get_params()
        get_params["node_num"] = self.node_num
        get_params["init_state"] = self.init_state
        get_params["goal_nodes"] = self.goal_nodes
        get_params["has_door_nodes"] = self.has_door_nodes
        get_params["cur_state"] = self.cur_state
        get_params["is_goal_terminal"] = self.is_goal_terminal
        get_params["success_rate"] = self.success_rate
        return get_params

    def get_neighbor(self, node):
        return list(self.G[node])

    def get_actions(self):
        actions = list()
        neighbor = self.get_neighbor(self.cur_state)
        neighbor_id = [node.id for node in neighbor]
        for a in ACTIONS:
            if a == "goto":
                for n in neighbor_id:
                    if not self.nodes[n].has_door():
                        actions.append((a, n))
            elif a == "approach":
                for n in neighbor_id:
                    if self.nodes[n].has_door():
                        actions.append((a, n))
            else:
                if self.cur_state.has_door():
                    actions.append((a, self.cur_state.id))
        return actions

    # Setter

    def set_nodes(self):
        nodes = [MDPGraphWorldNode(i, is_terminal=False) for i in range(self.node_num)]
        for i in self.success_rate:
            nodes[i].set_slip_prob(self.success_rate[i])
        for i in self.has_door_nodes:
            nodes[i].set_door(True, self.door_id[i], self.door_open_nodes[i])

        if self.is_goal_terminal:
            for i in self.goal_nodes:
                nodes[i].set_terminal(True)

        return nodes

    def set_graph(self):
        node = self.nodes
        graph_dist = {node[0]: [node[1], node[2]],
                      node[1]: [node[0], node[2], node[14]],
                      node[2]: [node[0], node[1], node[5]],
                      node[3]: [node[4]],
                      node[4]: [node[3], node[6]],
                      node[5]: [node[2], node[6], node[7], node[8]],
                      node[6]: [node[4], node[5], node[7], node[8]],
                      node[7]: [node[5], node[6], node[8]],
                      node[8]: [node[5], node[6], node[7], node[9]],
                      node[9]: [node[8], node[10], node[11]],
                      node[10]: [node[9], node[11]],
                      node[11]: [node[9], node[10], node[12]],
                      node[12]: [node[11], node[13], node[14]],
                      node[13]: [node[12], node[14]],
                      node[14]: [node[1], node[12], node[13]]}
        G = nx.Graph(graph_dist)
        nx.set_node_attributes(G, 0, "count")
        return G

    # Core

    def _transition_func(self, state, action):
        """
        transition function. it returns next state
        :param state: <State>
        :param action: <tuple <str, id>> action discription and node id
        :return: next_state <State>
        """
        self.G.nodes[state]['count'] += 1

        # print(self.get_neighbor(state))

        if state.is_terminal():
            return state

        if state.success_rate < random.random() and action[0] == "gothrough":
            action = ("fail", action[1])

        next_state = state

        if action[0] == "opendoor" and state.has_door() and not state.door_open():
            state.set_door(state.has_door(), state.get_door_id(), True)
            next_state = self.nodes[action[1]]
        elif action[0] == "gothrough" and state.has_door() and state.door_open():
            for node in self.get_neighbor(state):
                if node.get_door_id() == state.get_door_id():
                    next_state = node
        elif action[0] == "approach":
            next_state = self.nodes[action[1]]
            if next_state.get_door_id() == state.get_door_id():
                next_state = state
        elif action[0] == "goto":
            next_state = self.nodes[action[1]]
        else:
            next_state = state
            action = ("fail", action[1])

        # print(action)

        return next_state

    def _reward_func(self, state, action, next_state):
        """
        return rewards in next_state after taking action in state
        :param state: <State>
        :param action: <str>
        :param next_state: <State>
        :return: reward <float>
        """
        if next_state.is_terminal():
            return 10 - self.step_cost
        else:
            return 0 - self.step_cost

    def print_graph(self):
        import matplotlib.pyplot as plt
        pos = nx.spring_layout(self.G)
        nx.draw_networkx_nodes(self.G, pos, alpha=0.9, node_size=500)
        nodelist = [self.nodes[0], self.nodes[10]]
        nx.draw_networkx_nodes(self.G, pos, nodelist=nodelist, node_color='r', alpha=0.9,
                               node_size=500)
        nx.draw_networkx_labels(self.G, pos)
        nx.draw_networkx_edges(self.G, pos)
        plt.show()

    def save_graph_fig(self, filename="graph.png"):
        import matplotlib.pyplot as plt
        fix, ax = plt.subplots()
        pos = nx.spring_layout(self.G)
        nx.draw_networkx_nodes(self.G, pos, alpha=0.9, node_size=500)
        nodelist = [self.nodes[0], self.nodes[10]]
        nx.draw_networkx_nodes(self.G, pos, nodelist=nodelist, node_color='r', alpha=0.9,
                               node_size=500)
        nx.draw_networkx_labels(self.G, pos)
        nx.draw_networkx_edges(self.G, pos)
        plt.savefig(filename)
        del plt

    def save_graph(self, filename="graph.p"):
        with open(filename, "wb") as f:
            nx.write_gpickle(self.G, f)


class MDPGraphWorldNode(MDPStateClass):
    def __init__(self, id, is_terminal=False, has_door=False, door_id=None, door_open=False, success_rate=0.0):
        """
        A state in MDP
        :param id: <str>
        :param is_terminal: <bool>
        :param has_door: <bool>
        :param success_rate: <float>
        """
        self.id = id
        self.success_rate = success_rate
        self._has_door = has_door
        self._door_open = door_open
        self._door_id = door_id
        super().__init__(data=self.id, is_terminal=is_terminal)

    def __hash__(self):
        return hash(self.data)

    def __str__(self):
        return "s{0}".format(self.id)

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        assert isinstance(other, MDPGraphWorldNode), "Arg object is not in" + type(self).__module__
        return self.id == other.id

    def get_param(self):
        params_dict = dict()
        params_dict["id"] = self.id
        params_dict["success_rate"] = self.success_rate
        params_dict["has_door"] = self._has_door
        params_dict["door_open"] = self._door_open
        params_dict["door_id"] = self._door_id
        return params_dict

    def get_slip_prob(self):
        return self.success_rate

    def get_door_id(self):
        return self._door_id

    def set_slip_prob(self, new_slip_prob):
        self.success_rate = new_slip_prob

    def has_door(self):
        return self._has_door

    def door_open(self):
        return self._door_open

    def set_door(self, has_door, door_id, door_open):
        self._has_door = has_door
        self._door_id = door_id
        self._door_open = door_open


if __name__ == "__main__":
    Graph_world = MDPGraphWorld()
    Graph_world.reset()
    observation = Graph_world
    # Graph_world.print_graph()
    for t in range(50):
        # print(Graph_world.get_actions())
        random_action = (random.choice(Graph_world.get_actions()))
        print(observation.get_cur_state(), random_action, end=" ")
        observation, reward, done, info = Graph_world.step(random_action)
        print(observation.get_params())
        if done:
            print("Goal!")
            break
