# -*- coding: utf-8 -*-
from mdp.base.mdpBase import MDPBasisClass, MDPStateClass

from collections import namedtuple
import numpy as np
import random

ACTIONS = ["north", "south", "west", "east", "opendoor"]

MAP = (
    "+-----+",
    "| | : |",
    "| ; : |",
    "|S| :G|",
    "+-----+"
)

MAP2 = (
    "+-----------------+",
    "| : : | : : | :F: |",
    "| : : | : : ; : : |",
    "| : : ; : : | : : |",
    "| : : | : : | : : |",
    "| :S: | : : ; :G: |",
    "| : : | : : | : : |",
    "| : : ; : : | : : |",
    "| : : | : : ; : : |",
    "| : : | : : | : : |",
    "+-----------------+"
)


class GridWorld(MDPBasisClass):
    def __init__(self,
                 name="gridworld",
                 num_rows=3,
                 num_columns=3,
                 init_loc=(0, 0),
                 goal_loc=(7, 0),
                 door_loc=((0, 1),),
                 exit_flag=True,
                 gridmap=MAP2,
                 step_cost=0,
                 goal_reward=1):

        self.name = name
        self.num_rows = num_rows
        self.num_columns = num_columns
        self.init_loc = init_loc
        self.goal_loc = goal_loc
        self.door_loc = {"D_{0}{1}".format(i[0], i[1]): 0 for i in door_loc}

        if gridmap is not None:
            self.grid = self.make_gridworld(gridmap)
        else:
            self.grid = self.convert_gridworld()

        self.num_doors = len(door_loc)
        # row * columns * (open/close) ** doors
        self.number_of_states = self.num_rows * self.num_columns * 2 ** self.num_doors
        self.step_cost = step_cost
        self.goal_reward = goal_reward

        self.init_state = GridWorldState(self.init_loc[0], self.init_loc[1], self.door_loc, is_terminal=False)
        self.exit_flag = exit_flag
        super().__init__(self.init_state, self._transition_func, self._reward_func, ACTIONS)

    def __str__(self):
        return self.name + "{}x{}".format(self.num_rows, self.num_columns)

    def __repr__(self):
        return self.__str__()

    # Accessors

    @staticmethod
    def get_door_key(x, y):
        return "D_{0}{1}".format(x, y)

    def get_params(self):
        get_params = super().get_params()
        get_params["EnvName"] = self.name
        get_params["init_state"] = self.get_init_state()
        get_params["init_loc"] = self.init_loc
        get_params["goal_loc"] = self.goal_loc
        get_params["step_cost"] = self.step_cost
        get_params["is_goal_terminal"] = self.exit_flag
        get_params["goal_reward"] = self.goal_reward
        get_params["gridworld"] = self.grid
        return get_params

    def get_goal_reward(self):
        return self.goal_reward

    def get_goal_loc(self):
        return self.goal_loc

    def get_doors_loc(self):
        return self.door_loc

    def get_executable_actions(self, state):
        return self.get_actions()

    # Setter

    def set_step_cost(self, new_step_cost):
        self.step_cost = new_step_cost

    def set_goal_reward(self, new_goal_reward):
        self.goal_reward = new_goal_reward

    def set_init_loc(self, new_init_loc):
        self.init_loc = new_init_loc

    def set_goal_loc(self, new_goal_loc):
        self.goal_loc = new_goal_loc

    def set_door_locs(self, new_door_locs):
        self.door_loc = {"D_{0}{1}".format(i[0], i[1]): 0 for i in tuple(new_door_locs)}

    # Core

    def reset(self):
        return super().reset()

    def _transition_func(self, state, action):
        """
        transition function. it returns next state
        :param state: <State>
        :param action: <str>
        :return: next_state <State>
        """

        if action not in self.get_actions():
            raise Exception("Illegal action!: {} is not in {}".format(action, self.get_actions()))

        if state.is_terminal():
            return state

        x, y, d_list = state.get_data()

        if state.success_rate < random.random():
            # print("slip action: ")
            action = random.choice(self.get_actions())

        if action == "north":
            next_state = GridWorldState(x, min(y + 1, self.num_rows - 1), self.door_loc, is_terminal=False)

        elif action == "south":
            next_state = GridWorldState(x, max(y - 1, 0), self.door_loc, is_terminal=False)

        elif action == "east" and (self.grid[self.num_rows - y, 2 * x + 2] == b":" or
                                   (self.grid[self.num_rows - y, 2 * x + 2] == b";" and self.door_loc[self.get_door_key(x, y)])):
            next_state = GridWorldState(min(x + 1, self.num_columns - 1), y, self.door_loc, is_terminal=False)

        elif action == "west" and (self.grid[self.num_rows - y, 2 * x] == b":" or
                                   (self.grid[self.num_rows - y, 2 * x] == b";" and self.door_loc[self.get_door_key(x - 1, y)])):
            next_state = GridWorldState(max(x - 1, 0), y, self.door_loc, is_terminal=False)

        elif action == "opendoor" and self.grid[self.num_rows - y, 2 * x + 2] == b";":
            self.door_loc[self.get_door_key(x, y)] = 1
            next_state = GridWorldState(x, y, self.door_loc, is_terminal=False)

        elif action == "opendoor" and self.grid[self.num_rows - y, 2 * x] == b";":
            self.door_loc[self.get_door_key(x - 1, y)] = 1
            next_state = GridWorldState(x, y, self.door_loc, is_terminal=False)

        else:
            next_state = GridWorldState(x, y, self.door_loc, is_terminal=False)

        if (next_state.x, next_state.y) == self.goal_loc and self.exit_flag:
            next_state.set_terminal(True)

        # print(next_state)

        return next_state

    def _reward_func(self, state, action, next_state):
        """
        return rewards in next_state after taking action in state
        :param state: <State>
        :param action: <str>
        :param state: <State>
        :return: reward <float>
        """
        if state.is_terminal():
            return self.goal_reward
        else:
            return 0 - self.step_cost

    def make_gridworld(self, gridmap):
        self.num_rows = len(gridmap) - 2
        self.num_columns = len(gridmap[0]) - 2
        door_loc = list()
        grid = np.asarray(gridmap, dtype='c')
        for ybar, line in enumerate(gridmap):
            y = len(gridmap) - ybar - 2
            for x, c in enumerate(line):
                if c == 'S':
                    self.init_loc = (int(x / 2), y)
                elif c == 'G':
                    self.goal_loc = (int(x / 2), y)
                elif c == ';':
                    door_loc.append((int(x / 2) - 1, y))
        self.set_door_locs(door_loc)
        return grid

    def convert_gridworld(self):
        grid = [[' ' for y in range(self.num_rows)] for x in range(self.num_columns)]

    def print_gird(self):
        x, y, d_list = self.get_current_state()
        # print(x, y)
        grid = []
        for line in self.grid:
            tmp = []
            for c in line:
                tmp.append(c.decode())
            grid.append(tmp)
            # print("".join(list(line)))
        grid[self.num_rows - y][2 * x + 1] = "*"
        for line in grid:
            print("".join(line))


class GridWorldState(MDPStateClass):
    def __init__(self, x, y, d_dict, is_terminal=False):
        self.x = x
        self.y = y
        self.doors = self.door_namedtuple(d_dict)
        self.success_rate = 0.95
        super().__init__(data=(self.x, self.y, self.doors), is_terminal=is_terminal)

    def __str__(self):
        return "pos({0}, {1}, {2})".format(self.x, self.y, self.doors)

    def __repr__(self):
        return self.__str__()

    def set_successrate(self, success_rate):
        self.success_rate = success_rate

    def door_namedtuple(self, v):
        if isinstance(v, dict):
            return namedtuple('D', v.keys())(**{x: self.door_namedtuple(y) for x, y in v.items()})
        if isinstance(v, (list, tuple)):
            return [self.door_namedtuple(x) for x in v]
        return v


if __name__ == "__main__":
    ###########################
    # GET THE GRIDWORLD
    ###########################
    env = GridWorld()
    observation = env.reset()
    random_action = random.choice(ACTIONS)
    env.print_gird()

    for t in range(500):
        # if random_action == "opendoor":
        env.print_gird()
        print(observation, env.get_state_count(observation), random_action)
        observation, reward, done, info = env.step(random_action)
        random_action = random.choice(ACTIONS)
        if done:
            print("The agent arrived at tearminal state.")
            # print(observation.get_params())
            print("Exit")
            exit()
