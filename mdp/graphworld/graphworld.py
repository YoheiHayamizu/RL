from mdp.base.mdpBase import MDPBasisClass, MDPStateClass
from mdp.graphworld.config import *
import random
from collections import defaultdict
import itertools
import networkx as nx
import json


class GraphWorld(MDPBasisClass):
    def __init__(self,
                 name="graphworld",
                 node_num=node_num,
                 init_node=0,
                 goal_node=17,
                 node_has_door=(),
                 graphmap_path="map.json",
                 exit_flag=True,
                 step_cost=1.0,
                 goal_reward=1,
                 stack_cost=50
                 ):
        self.name = name
        self.node_num = node_num
        self.init_node = init_node
        self.goal_node = goal_node
        self.nodes_has_door = node_has_door

        if graphmap_path is not None:
            self.graph, self.G = self.make_graph(graphmap_path)
        else:
            self.graph, self.G = self.convert_graphworld()

        self.num_doors = len(node_has_door)
        self.number_of_states = (node_num - int(len(node_has_door) / 2)) * 2 ** len(node_has_door)
        self.goal_reward = goal_reward
        self.step_cost = step_cost
        self.stack_cost = stack_cost

        self.init_state = GraphWorldState(self.graph[self.init_node]['node_id'],
                                          self.graph[self.init_node]['door_id'],
                                          self.graph[self.init_node]['door_open'],
                                          self.graph[self.init_node]['success_rate'],
                                          self.graph[self.init_node]['stack_rate'])
        self.exit_flag = exit_flag
        super().__init__(self.init_state, self._transition_func, self._reward_func, self.get_actions())

    def __str__(self):
        return self.name + "_n-" + str(self.node_num)

    def __repr__(self):
        return self.__str__()

    # Accessors

    def get_params(self):
        get_params = super().get_params()
        get_params["EnvName"] = self.name
        get_params["init_state"] = self.get_init_state()
        get_params["node_num"] = self.node_num
        get_params["init_node"] = self.init_node
        get_params["goal_node"] = self.goal_node
        get_params["is_goal_terminal"] = self.exit_flag
        get_params["goal_reward"] = self.goal_reward
        get_params["step_cost"] = self.step_cost
        get_params["stack_cost"] = self.stack_cost
        return get_params

    def get_neighbor(self, node):
        return list(self.graph[node])

    def get_actions(self):
        actions = defaultdict(lambda: set())
        for node in self.graph.keys():
            node_id, door_id, door_open, success_rate, stack_rate, adjacent = self.graph[node].values()
            for action in ["goto", "approach", "opendoor", "gothrough"]:
                for n in adjacent + [node_id]:
                    actions[GraphWorldState(node_id, door_id, False)].add((action, n))
                    actions[GraphWorldState(node_id, door_id, True)].add((action, n))

        #     for action in ["goto", "approach", "opendoor", "gothrough"]:
        #         if action == "goto":
        #             for n in adjacent:
        #                 if self.graph[n]['door_id'] is None:
        #                     actions[GraphWorldState(node_id, door_id, False)].add((action, n))
        #                     actions[GraphWorldState(node_id, door_id, True)].add((action, n))
        #         elif action == "approach":
        #             for n in adjacent:
        #                 if self.graph[n]['door_id'] is not None and door_id != self.graph[n]['door_id']:
        #                     actions[GraphWorldState(node_id, door_id, False)].add((action, n))
        #                     actions[GraphWorldState(node_id, door_id, True)].add((action, n))
        #         elif door_id is not None and (action == "opendoor" or action == "gothrough"):
        #             actions[GraphWorldState(node_id, door_id, False)].add((action, node_id))
        #             actions[GraphWorldState(node_id, door_id, True)].add((action, node_id))
        # for node, action in actions.items():
        #     print(node, action)
        return actions

    def get_stack_cost(self):
        return self.stack_cost

    def get_goal_reward(self):
        return self.goal_reward

    def get_executable_actions(self, state):
        return self.get_actions()[state]

    # Setter

    def make_graph(self, graphmap):
        with open(graphmap, 'r') as f:
            data = json.load(f)
        self.name = data['name']
        graph = {}
        graph_dict = {}
        for datum in data['info']:
            graph[datum['node_id']] = datum
            graph_dict[datum['node_id']] = datum['adjacent']
        G = nx.Graph(graph_dict)
        return graph, G

    # Core

    def _is_goal_state(self, state):
        return state.id == self.goal_node

    def _transition_func(self, state, action):
        """
        transition function. it returns next state
        :param state: <State>
        :param action: <tuple <str, id>> action discription and node id
        :return: next_state <State>
        """
        print(state, action)
        if action not in self.get_executable_actions(state):
            raise Exception("Illegal action!: {} is not in {}".format(action, self.get_executable_actions(state)))

        if state.is_terminal():
            return state

        rand = random.random()
        if state.success_rate[0] < rand and not self._is_goal_state(state):
            if action[0] == "gothrough":
                action = ("fail", action[1])

            if action[0] == "opendoor":
                action = ("fail", action[1])

            if action[0] == "approach":
                miss = random.choice(self.get_neighbor(state) + [state])
                action = ("approach", miss.id)

            if action[0] == "goto":
                miss = random.choice(self.get_neighbor(state) + [state])
                action = ("goto", miss.id)

        next_state = state

        if action[0] == "opendoor" and state == self.nodes[action[1]] and state.has_door():
            state.set_door(state.has_door(), state.get_door_id(), True)
            next_state = state
        elif action[0] == "gothrough" and state.has_door() and state.get_door_state():
            for node in self.get_neighbor(state):
                if node.get_door_id() == state.get_door_id():
                    next_state = node
            next_state.set_door(state.has_door(), state.get_door_id(), True)
        elif action[0] == "approach" and self.nodes[action[1]].has_door():
            next_state = self.nodes[action[1]]
            if next_state.get_door_id() == state.get_door_id():
                next_state = state
        elif action[0] == "goto" and not self.nodes[action[1]].has_door():
            next_state = self.nodes[action[1]]
        else:
            next_state = state
            action = ("fail", action[1])

        if next_state.success_rate[0] + next_state.success_rate[1] < rand and \
                (action[0] == "gothrough" or action[0] == "opendoor" or action[0] == "fail") and \
                not self._is_goal_state(next_state):
            next_state.is_stack = True
            next_state.set_terminal()

        return next_state

    def _reward_func(self, state, action, next_state):
        """
        return rewards in next_state after taking action in state
        :param state: <State>
        :param action: <str>
        :param next_state: <State>
        :return: reward <float>
        """
        if self._is_goal_state(state):
            return self.get_goal_reward()
        elif next_state.get_is_stack():
            return -self.get_stack_cost()
        else:
            return 0 - self.step_cost

    def reset(self):
        return super().reset()

    def print_graph(self):
        import matplotlib.pyplot as plt
        pos = nx.spring_layout(self.graph)
        nx.draw_networkx_nodes(self.graph, pos, alpha=0.9, node_size=500)
        nodelist = [self.nodes[0], self.nodes[10]]
        nx.draw_networkx_nodes(self.graph, pos, nodelist=nodelist, node_color='r', alpha=0.9,
                               node_size=500)
        nx.draw_networkx_labels(self.graph, pos)
        nx.draw_networkx_edges(self.graph, pos)
        plt.show()

    def save_graph_fig(self, filename="graph.png"):
        import matplotlib.pyplot as plt
        fix, ax = plt.subplots()
        pos = nx.spring_layout(self.graph)
        nx.draw_networkx_nodes(self.graph, pos, alpha=0.9, node_size=500)
        nodelist = [self.nodes[0], self.nodes[10]]
        nx.draw_networkx_nodes(self.graph, pos, nodelist=nodelist, node_color='r', alpha=0.9,
                               node_size=500)
        nx.draw_networkx_labels(self.graph, pos)
        nx.draw_networkx_edges(self.graph, pos)
        plt.savefig(filename)
        del plt

    def save_graph(self, filename="graph.p"):
        with open(filename, "wb") as f:
            nx.write_gpickle(self.graph, f)


class GraphWorldState(MDPStateClass):
    def __init__(self, node_id, door_id=None, door_open=None, success_rate=1.0, stack_rate=0.0,
                 is_terminal=False):
        """
        A state in MDP
        :param node_id: <str>
        :param is_terminal: <bool>
        :param has_door: <bool>
        :param success_rate: <float>
        """
        self.node_id = node_id
        self._door_id = door_id
        if door_id is not None:
            self._door_open = door_open
        else:
            self._door_open = None
        self.is_stack = False
        self.success_rate = success_rate
        self.stack_rate = stack_rate
        super().__init__(data=(self.node_id, self._door_id, self._door_open), is_terminal=is_terminal)

    def __hash__(self):
        return hash(self.data)

    def __str__(self):
        if self.has_door():
            return "s{0}_d{1}_{2}".format(self.node_id, self._door_id, self._door_open)
        else:
            return "s{0}".format(self.node_id)

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        assert isinstance(other, GraphWorldState), "Arg object is not in" + type(self).__module__
        return self.node_id == other.node_id

    def get_param(self):
        params_dict = dict()
        params_dict["id"] = self.node_id
        params_dict["door_open"] = self._door_open
        params_dict["door_id"] = self._door_id
        params_dict["success_rate"] = self.success_rate
        return params_dict

    def get_success_rate(self):
        return self.success_rate

    def get_door_id(self):
        return self._door_id

    def get_door_state(self):
        if self.has_door():
            return self._door_open
        else:
            raise Exception("This state does not have a door.")

    def get_is_stack(self):
        return self.is_stack

    def has_door(self):
        if self._door_id is not None:
            return True
        return False

    def set_success_rate(self, new_success_rate):
        self.success_rate = new_success_rate

    def set_door(self, door_id, door_open):
        self._door_id = door_id
        self._door_open = door_open


if __name__ == "__main__":
    ###########################
    # GET THE GRIDWORLD
    ###########################
    env = GraphWorld()
    observation = env.reset()
    ACTIONS = env.get_executable_actions(observation)
    print(ACTIONS)
    random_action = random.sample(ACTIONS, 1)[0]

    for t in range(500):
        # if random_action == "opendoor":
        print(observation, env.get_state_count(observation), random_action)
        observation, reward, done, info = env.step(random_action)
        random_action = random.choice(ACTIONS)
        if done:
            print("The agent arrived at tearminal state.")
            # print(observation.get_params())
            print("Exit")
            exit()
