from mdp.base.mdpBase import MDPBasisClass, MDPStateClass
from mdp.graphworld.config import *
import random
from collections import defaultdict
import networkx as nx
import json


class GraphWorld(MDPBasisClass):
    def __init__(self,
                 name="graphworld",
                 node_num=19,
                 init_node=0,
                 goal_node=17,
                 node_has_door=(),
                 graphmap_path=MAP_PATH + "map.json",
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

    def get_adjacent(self, node_id):
        return self.graph[node_id]['adjacent']

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

    def get_executable_actions(self, state=None):
        if state is None:
            return self.get_executable_actions(self.init_state)
        return list(self.get_actions()[state])

    # Setter

    def set_goal_reward(self, new_goal_reward):
        self.goal_reward = new_goal_reward

    def set_step_cost(self, new_step_cost):
        self.step_cost = new_step_cost

    def set_stack_cost(self, new_stack_cost):
        self.stack_cost = new_stack_cost

    def make_graph(self, graphmap):
        with open(graphmap, 'r') as f:
            data = json.load(f)
        graph = {}
        graph_dict = {}
        for datum in data['info']:
            graph[datum['node_id']] = datum
            graph_dict[datum['node_id']] = datum['adjacent']
        self.name = data['name']
        self.node_num = len(graph_dict)
        G = nx.Graph(graph_dict)
        return graph, G

    def convert_graphworld(self):
        raise NotImplementedError

    # Core

    def is_goal_state(self, state):
        return state.get_node_id() == self.goal_node

    def _transition_func(self, state, action):
        """
        transition function. it returns next state
        :param state: <State>
        :param action: <tuple <str, id>> action discription and node id
        :return: next_state <State>
        """
        if action not in self.get_executable_actions(state):
            raise Exception("Illegal action!: {} is not in {}".format(action, self.get_executable_actions(state)))

        if state.is_terminal():
            return state

        rand = random.random()
        if state.success_rate + state.stack_rate < rand and not self.is_goal_state(state):
            if action[0] == "gothrough":
                action = ("fail", action[1])

            if action[0] == "opendoor":
                action = ("fail", action[1])

            if action[0] == "approach":
                miss = random.choice(self.get_adjacent(state.get_node_id()))
                action = ("approach", miss)

            if action[0] == "goto":
                miss = random.choice(self.get_adjacent(state.get_node_id()))
                action = ("goto", miss)

        next_state = state

        a, n = action
        if a == "opendoor" and state.get_node_id() == self.graph[n]['node_id'] and state.has_door():
            self.graph[n]['door_open'] = True
            next_state = GraphWorldState(n, self.graph[n]['door_id'], self.graph[n]['door_open'],
                                         self.graph[n]['success_rate'], self.graph[n]['stack_rate'])
        elif a == "gothrough" and state.has_door() and state.get_door_state():
            for node in self.graph[n]['adjacent']:
                if self.graph[node]['door_id'] == state.get_door_id():
                    next_state = GraphWorldState(node, self.graph[node]['door_id'], self.graph[node]['door_open'],
                                                 self.graph[node]['success_rate'], self.graph[node]['stack_rate'])
        elif a == "approach" and self.graph[n]['door_id'] is not None and state.get_door_id() != self.graph[n][
            'door_id']:
            next_state = GraphWorldState(n, self.graph[n]['door_id'], self.graph[n]['door_open'],
                                         self.graph[n]['success_rate'], self.graph[n]['stack_rate'])
        elif a == "goto" and self.graph[n]['door_id'] is None:
            next_state = GraphWorldState(n, self.graph[n]['door_id'], self.graph[n]['door_open'],
                                         self.graph[n]['success_rate'], self.graph[n]['stack_rate'])
        else:
            next_state = state
            action = ("fail", n)

        if next_state.success_rate < rand < next_state.success_rate + next_state.stack_rate and (
                a == "gothrough" or a == "opendoor" or a == "fail") and not self.is_goal_state(next_state):
            next_state.is_stack = True
            next_state.set_terminal(True)

        if self.is_goal_state(next_state) and self.exit_flag:
            next_state.set_terminal(True)

        return next_state

    def _reward_func(self, state, action, next_state):
        """
        return rewards in next_state after taking action in state
        :param state: <State>
        :param action: <str>
        :param next_state: <State>
        :return: reward <float>
        """
        if self.is_goal_state(state):
            return self.get_goal_reward()
        elif state.get_is_stack():
            return - self.get_stack_cost()
        else:
            return - self.step_cost

    def reset(self):
        return super().reset()

    def print_graph(self):
        import matplotlib.pyplot as plt
        pos = nx.spring_layout(self.G, pos={0: [-0.8540059, 0.98151456],
                                             1: [-0.85041632, 0.73686487],
                                             2: [-0.62514745, 0.8359094],
                                             3: [-0.76355886, 0.40821448],
                                             4: [-0.32916205, 0.59689845],
                                             5: [-0.49406721, 0.11968062],
                                             6: [-0.49863244, -0.28250672],
                                             7: [-0.28264265, -0.58982159],
                                             8: [-0.09594634, 0.2363157],
                                             9: [-0.07191173, -0.08622657],
                                             10: [-0.0169109, -0.41152061],
                                             11: [0.39693677, 0.13289499],
                                             12: [0.5717122, -0.11621789],
                                             13: [0.62289137, -0.4189907],
                                             14: [0.7954032, 0.02493817],
                                             15: [1., -0.26606676],
                                             16: [0.87920069, -0.57667969],
                                             17: [0.09609302, -0.66380614],
                                             18: [0.52016459, -0.66139458]})
        nx.draw_networkx_nodes(self.G, pos, node_color='black', alpha=0.7, node_size=400)
        nx.draw_networkx_edges(self.G, pos)
        nx.draw_networkx_nodes(self.G, pos, nodelist=[self.init_node], node_color='b', alpha=0.9, node_size=400)
        nx.draw_networkx_nodes(self.G, pos, nodelist=[self.goal_node], node_color='r', alpha=0.9, node_size=400)
        nx.draw_networkx_nodes(self.G, pos, nodelist=[self.get_current_state().get_node_id()], node_color='yellow',
                               alpha=0.9, node_size=200)
        nx.draw_networkx_labels(self.G, pos, font_color='white')
        plt.pause(0.01)
        # plt.show()
        plt.clf()

    def save_graph_fig(self, filename="graph.png"):
        import matplotlib.pyplot as plt
        fix, ax = plt.subplots()
        pos = nx.spring_layout(self.G, pos={0: [-0.8540059, 0.98151456],
                                             1: [-0.85041632, 0.73686487],
                                             2: [-0.62514745, 0.8359094],
                                             3: [-0.76355886, 0.40821448],
                                             4: [-0.32916205, 0.59689845],
                                             5: [-0.49406721, 0.11968062],
                                             6: [-0.49863244, -0.28250672],
                                             7: [-0.28264265, -0.58982159],
                                             8: [-0.09594634, 0.2363157],
                                             9: [-0.07191173, -0.08622657],
                                             10: [-0.0169109, -0.41152061],
                                             11: [0.39693677, 0.13289499],
                                             12: [0.5717122, -0.11621789],
                                             13: [0.62289137, -0.4189907],
                                             14: [0.7954032, 0.02493817],
                                             15: [1., -0.26606676],
                                             16: [0.87920069, -0.57667969],
                                             17: [0.09609302, -0.66380614],
                                             18: [0.52016459, -0.66139458]})
        nx.draw_networkx_nodes(self.G, pos, node_color='black', alpha=0.7, node_size=400)
        nx.draw_networkx_edges(self.G, pos)
        nx.draw_networkx_nodes(self.G, pos, nodelist=[self.init_node], node_color='b', alpha=0.9, node_size=400)
        nx.draw_networkx_nodes(self.G, pos, nodelist=[self.goal_node], node_color='r', alpha=0.9, node_size=400)
        nx.draw_networkx_nodes(self.G, pos, nodelist=[self.get_current_state().get_node_id()], node_color='yellow',
                               alpha=0.9, node_size=200)
        nx.draw_networkx_labels(self.G, pos, font_color='white')
        plt.show()
        plt.savefig(filename)
        del plt

    def save_graph(self, filename="graph.p"):
        with open(filename, "wb") as f:
            nx.write_gpickle(self.graph, f)


class GraphWorldState(MDPStateClass):
    def __init__(self, node_id, door_id=None, door_open=None, success_rate=1.0, stack_rate=0.0,
                 is_terminal=False):
        """ Inheritance of MDPStateClass for graphworld

        :param node_id:
        :param door_id:
        :param door_open:
        :param success_rate:
        :param stack_rate:
        :param is_terminal:
        """
        self.node_id = node_id
        self.door_id = door_id
        if door_id is not None:
            self._door_open = door_open
        else:
            self._door_open = None
        self.is_stack = False
        self.success_rate = success_rate
        self.stack_rate = stack_rate
        super().__init__(data=(self.node_id, self.door_id, self._door_open), is_terminal=is_terminal)

    def __hash__(self):
        return hash(self.data)

    def __str__(self):
        if self.has_door():
            return "s{0}_d{1}_{2}".format(self.node_id, self.door_id, self._door_open)
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
        params_dict["door_id"] = self.door_id
        params_dict["success_rate"] = self.success_rate
        return params_dict

    def get_success_rate(self):
        return self.success_rate

    def get_node_id(self):
        return self.node_id

    def get_door_id(self):
        return self.door_id

    def get_door_state(self):
        if self.has_door():
            return self._door_open
        else:
            raise Exception("This state does not have a door.")

    def get_is_stack(self):
        return self.is_stack

    def has_door(self):
        if self.door_id is not None:
            return True
        return False

    def set_success_rate(self, new_success_rate):
        self.success_rate = new_success_rate

    def set_stack_rate(self, new_stack_rate):
        self.new_stack_rate = new_stack_rate


if __name__ == "__main__":
    ###########################
    # GET THE GRIDWORLD
    ###########################
    env = GraphWorld()
    observation = env.reset()
    ACTIONS = env.get_executable_actions(observation)
    random_action = random.sample(ACTIONS, 1)[0]

    for t in range(500):
        print(observation, env.get_state_count(observation), random_action)
        observation, reward, done, info = env.step(random_action)
        ACTIONS = env.get_executable_actions(observation)
        random_action = random.sample(ACTIONS, 1)[0]
        env.print_graph()
        if done:
            env.save_graph_fig()
            print("The agent arrived at tearminal state.")
            # print(observation.get_params())
            print("Exit")
            exit()
