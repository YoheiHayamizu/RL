from mdp.MDPBasis import MDPBasisClass
from mdp.MDPState import MDPStateClass
from mdp.GraphWorldConstants import *
import random
import networkx as nx


class MDPGraphWorld(MDPBasisClass):
    def __init__(self, node_num=15,
                 goal_nodes=goal_nodes_tuple, has_door_nodes=has_door_nodes_tuple,
                 is_goal_terminal=True, failure_prob=failure_prob_dict1, step_cost=1.0, name="Graphworld"
                 ):
        self.actions = ACTIONS
        self.node_num = node_num
        self.is_goal_terminal = is_goal_terminal
        self.failure_prob = failure_prob
        self.goal_nodes = goal_nodes
        self.has_door_nodes = has_door_nodes
        self.step_cost = step_cost
        self.name = name
        self.nodes = self.set_nodes()
        self.init_state = self.nodes[0]
        self.cur_state = self.nodes[0]
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
        get_params["failure_prob"] = self.failure_prob
        return get_params

    def get_neighbor(self, node):
        return list(self.G[node])

    def get_actions(self):
        actions = list()
        neighbor = self.get_neighbor(self.cur_state)
        neighbor_id = [node.id for node in neighbor]
        for a in ACTIONS:
            if a in ["goto", "approach"]:
                for n in neighbor_id:
                    actions.append((a, n))
            else:
                actions.append((a, self.cur_state.id))
        return actions

    # Setter

    def set_nodes(self):
        nodes = [MDPGraphWorldNode(i, is_terminal=False) for i in range(self.node_num)]
        for i in self.failure_prob:
            nodes[i].set_slip_prob(self.failure_prob[i])
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
        return G

    # Core

    def _transition_func(self, state, action):
        """
        transition function. it returns next state
        :param state: <State>
        :param action: <tuple <str, id>> action discription and node id
        :return: next_state <State>
        """

        if state.is_terminal():
            return state

        if state.failure_prob > random.random() and action[0] == "gothrough":
            action = random.choice([("wait", n) for n in [node.id for node in self.get_neighbor(state)]])

        if action[0] not in ACTIONS:
            print("the action is not defined in transition function")
            next_state = state
        elif action[0] == "goto" and self.nodes[action[1]] in self.get_neighbor(state):
            next_state = self.nodes[action[1]]
        elif action[0] == "approach" and self.nodes[action[1]] in self.get_neighbor(state):
            next_state = self.nodes[action[1]]
        elif action[0] == "opendoor" and self.nodes[action[1]] in self.get_neighbor(state) and state.has_door():
            next_state = self.nodes[action[1]]
        elif action[0] == "gothrough" and self.nodes[action[1]] in self.get_neighbor(state):
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
        nx.draw(self.G)


class MDPGraphWorldNode(MDPStateClass):
    def __init__(self, id, is_terminal=False, has_door=False, failure_prob=0.0):
        """
        A state in MDP
        :param id: <str>
        :param is_terminal: <bool>
        :param has_door: <bool>
        :param failure_prob: <float>
        """
        self.id = id
        self.failure_prob = failure_prob
        self._has_door = has_door
        super().__init__(data=(self.id, self.failure_prob, self._has_door), is_terminal=is_terminal)

    def __hash__(self):
        return hash(tuple(self.data))

    def __str__(self):
        return "s{0}".format(self.id)

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        assert isinstance(other, MDPGraphWorldNode), "Arg object is not in" + type(self).__module__
        return self.id == other.id

    def get_slip_prob(self):
        return self.failure_prob

    def set_slip_prob(self, new_slip_prob):
        self.failure_prob = new_slip_prob

    def has_door(self):
        return self.failure_prob

    def set_door(self, has_door):
        self._has_door = has_door


if __name__ == "__main__":
    Graph_world = MDPGraphWorld()
    Graph_world.reset()
    observation = Graph_world
    print(Graph_world.get_actions())
    for t in range(100):
        random_action = (random.choice(Graph_world.get_actions()))
        print(observation.get_cur_state(), random_action)
        observation, reward, done, info = Graph_world.step(random_action)
        if done:
            print("Goal!")
            exit()
