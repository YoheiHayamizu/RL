from mdp.MDPBasis import MDPBasisClass
from mdp.MDPState import MDPStateClass
from mdp.GraphWorldConstants import *
import random
import networkx as nx


class MDPGraphWorld(MDPBasisClass):
    def __init__(self, node_num=15, init_node=0,
                 goal_nodes=goal_nodes_tuple, has_door_nodes=has_door_nodes_tuple,
                 is_goal_terminal=True, success_rate=success_rate_dict1, step_cost=1.0, name="Graphworld"
                 ):
        self.actions = ACTIONS
        self.node_num = node_num
        self.is_goal_terminal = is_goal_terminal
        self.success_rate = success_rate
        self.goal_nodes = goal_nodes
        self.has_door_nodes = has_door_nodes
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
            nodes[i].set_door(True)

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

        if state.is_terminal():
            return state

        if state.success_rate < random.random() and action[0] == "gothrough":
            action = ("wait", action[1])

        if action[0] not in self.actions:
            print("the action is not defined in transition function")
            next_state = state
        elif action[0] == "goto" and self.nodes[action[1]] in self.get_neighbor(state):
            next_state = self.nodes[action[1]]
        elif action[0] == "approach" and self.nodes[action[1]] in self.get_neighbor(state):
            next_state = self.nodes[action[1]]
        elif action[0] == "opendoor" and self.nodes[action[1]] in self.get_neighbor(state) and state.has_door():
            next_state = self.nodes[action[1]]
        elif action[0] == "gothrough" and self.nodes[action[1]] in self.get_neighbor(state) and state.has_door():
            next_state = self.nodes[action[1]]
        else:
            next_state = state

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


class MDPGraphWorldNode(MDPStateClass):
    def __init__(self, id, is_terminal=False, has_door=False, success_rate=0.0):
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

    def get_slip_prob(self):
        return self.success_rate

    def set_slip_prob(self, new_slip_prob):
        self.success_rate = new_slip_prob

    def has_door(self):
        return self._has_door

    def set_door(self, has_door):
        self._has_door = has_door


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
