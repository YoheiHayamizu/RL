from mdp.base.mdpBase import MDPBasisClass, MDPStateClass
import random

# import copy

ACTIONS = ["up", "down", "left", "right"]

MAP = [[' ', ' ', ' ', 'G'],
       ['#', '#', ' ', '#'],
       [' ', '#', ' ', ' '],
       [' ', '#', '#', ' '],
       ['S', ' ', ' ', ' ']]


class BlockWorld(MDPBasisClass):
    def __init__(self,
                 name="blockworld",
                 width=5,
                 height=5,
                 init_loc=(0, 0),
                 goal_loc=(4, 4),
                 walls_loc=((),),
                 holes_loc=((),),
                 exit_flag=True,
                 blockmap=None,
                 slip_prob=0.0,
                 step_cost=0.0,
                 hole_cost=1.0,
                 goal_reward=1
                 ):
        """
        Constructor of MDPMaze
        :param width: <int> width of maze
        :param height: <int> height of maze
        :param walls_loc: <tuple<tuple>> walls
        :param holes_loc: <tuple<tuple>> holes
        :param exit_flag: <bool> if true, episode will terminate when agent reaches goals_loc
        :param slip_prob: <float> probability of that agent actions fail
        :param step_cost: <float> cost of one action
        :param hole_cost: <float> cost of dropping a hole
        :param name: <str> name of maze
        """

        self.name = name
        self.width = width
        self.height = height
        self.start = init_loc
        self.goal = goal_loc
        self.walls = walls_loc
        self.holes = holes_loc
        self.slip_prob = slip_prob
        self.step_cost = step_cost
        self.hole_cost = hole_cost
        self.goal_reward = goal_reward

        if blockmap is not None:
            self.blocks = self.make_blockworld(blockmap)
        else:
            self.blocks = self.convert_blockworld()

        # self.init_grid = copy.deepcopy(self.grid)
        self.states = [[BlockWorldState(x, y) for y in range(len(self.blocks[x]))] for x in range(len(self.blocks))]
        self.init_state = self.states[self.start[0]][self.start[1]]
        self.exit_flag = exit_flag
        self.set_actions(ACTIONS)
        super().__init__(self.init_state, self._transition_func, self._reward_func, self.actions)

    def __str__(self):
        return self.name + "_h-" + str(self.height) + "_w-" + str(self.width)

    def __repr__(self):
        return self.__str__()

    # Accessors

    def get_params(self):
        get_params = super().get_params()
        get_params["current_state"] = self.get_current_state()
        get_params["init_state"] = self.get_init_state()
        get_params["is_goal_terminal"] = self.exit_flag
        get_params["slip_prob"] = self.slip_prob
        get_params["hole_cost"] = self.hole_cost
        get_params["blockworld"] = self.blocks
        return get_params

    def get_slip_prob(self):
        return self.slip_prob

    def get_hole_cost(self):
        return self.hole_cost

    def get_goal_reward(self):
        return self.goal_reward

    def get_goals_loc(self):
        return self.goal

    def get_holes_loc(self):
        return self.holes

    def get_state(self, x, y):
        return self.states[x][y]

    def get_states(self):
        flatten = lambda l: [item for sublist in l for item in sublist]
        return flatten(self.states)

    def get_actions(self, state=None):
        if state is None:
            return self.actions
        return self.actions[state]

    def get_visited(self, state):
        return self.blocks[state.x][state.y][1]

    def get_executable_actions(self, state):
        return self.actions[state]

    # Setter

    def set_slip_prob(self, new_slip_prob):
        self.slip_prob = new_slip_prob

    def set_step_cost(self, new_step_cost):
        self.step_cost = new_step_cost

    def set_hole_cost(self, new_hole_cost):
        self.hole_cost = new_hole_cost

    def set_goal_reward(self, new_goal_reward):
        self.goal_reward = new_goal_reward

    def set_init_loc(self, new_init_loc):
        self.start = new_init_loc

    def set_goal_locs(self, new_goal_locs):
        self.goal = new_goal_locs

    def set_wall_locs(self, new_wall_locs):
        self.walls = new_wall_locs

    def set_hole_locs(self, new_hole_locs):
        self.holes = new_hole_locs

    def add_goals_loc(self, new_goal_loc):
        self.goal.append(new_goal_loc)

    def add_holes_loc(self, new_hole_loc):
        self.holes.append(new_hole_loc)

    def set_actions(self, new_action_sets):
        self.actions = {state: new_action_sets for row in self.states for state in row}

    # Core

    def reset(self):
        # print(self.cur_state)
        self.get_current_state().set_terminal(False)
        return super().reset()

    def _transition_func(self, state, action):
        """
        transition function. it returns next state
        :param state: <State>
        :param action: <str>
        :return: next_state <State>
        """

        if action not in self.get_actions(state):
            raise Exception("Illegal action!")

        if state.is_terminal():
            return state

        if self.slip_prob > random.random():
            print("slip action: ")
            action = random.choice(self.get_actions(state))

        x, y = state.x, state.y

        if action == "up" and self.__is_allowed(x, y + 1) and not self.__is_wall(x, y + 1):
            next_state = self.get_state(x, y + 1)
        elif action == "down" and self.__is_allowed(x, y - 1) and not self.__is_wall(x, y - 1):
            next_state = self.get_state(x, y - 1)
        elif action == "left" and self.__is_allowed(x - 1, y) and not self.__is_wall(x - 1, y):
            next_state = self.get_state(x - 1, y)
        elif action == "right" and self.__is_allowed(x + 1, y) and not self.__is_wall(x + 1, y):
            next_state = self.get_state(x + 1, y)
        else:
            next_state = self.get_state(x, y)
        # print("current goal is {0}".format(self.goal_loc))
        if ((next_state.x, next_state.y) in self.holes or (
                next_state.x, next_state.y) == self.goal) and self.exit_flag:
            next_state.set_terminal(True)
        return next_state

    def __is_allowed(self, x, y):
        """
        return True if (x,y) is a valid location.
        :param x: <int>
        :param y: <int>
        :return: <bool>
        """
        if y < 0 or y >= self.height: return False
        if x < 0 or x >= self.width: return False
        return True

    def __is_wall(self, x, y):
        """
        return True if (x,y) is a wall location.
        :param x: <int>
        :param y: <int>
        :return: <bool>
        """
        if (x, y) in self.walls: print("hit wall!")
        return (x, y) in self.walls

    def _reward_func(self, state, action, next_state):
        """
        return rewards in next_state after taking action in state
        :param state: <State>
        :param action: <str>
        :param next_state: <State>
        :return: reward <float>
        """
        if (next_state.x, next_state.y) == self.goal:
            return self.get_goal_reward()
        elif (next_state.x, next_state.y) in self.holes:
            return -self.get_hole_cost()
        else:
            return 0 - self.step_cost

    def make_blockworld(self, grid_string):
        self.width, self.height = len(grid_string[0]), len(grid_string)
        new_goal_locs = list()
        new_hole_locs = list()
        new_wall_locs = list()
        new_door_locs = list()
        new_init_loc = None
        grid = [[[' ', 0] for y in range(self.height)] for x in range(self.width)]
        for ybar, line in enumerate(grid_string):
            y = self.height - ybar - 1
            for x, el in enumerate(line):
                grid[x][y][0] = el
                if el == 'G':
                    new_goal_locs.append((x, y))
                elif el == 'O':
                    new_hole_locs.append((x, y))
                elif el == '#':
                    new_wall_locs.append((x, y))
                elif el == 'D':
                    new_door_locs.append((x, y))
                elif el == 'S':
                    new_init_loc = (x, y)
        self.set_goal_locs(tuple(new_goal_locs))
        self.set_hole_cost(tuple(new_hole_locs))
        self.set_wall_locs(tuple(new_wall_locs))
        self.set_init_loc(new_init_loc)
        return grid

    def convert_blockworld(self):
        grid = [[' ' for y in range(self.height)] for x in range(self.width)]
        for x, y in self.holes:
            grid[x][y] = 'H'
        for el in self.walls:
            grid[el[0]][el[1]] = '#'
        x, y = self.start
        grid[x][y] = 'S'
        x, y = self.goal
        grid[x][y] = 'G'
        return grid

    def print_blockworld(self):
        x, y = self.get_current_state().x, self.get_current_state().y
        tmp = self.blocks[x][y]
        self.blocks[x][y] += '*'
        tmp2 = [[self.blocks[x][y] for x in range(self.width)] for y in range(self.height)]
        tmp2.reverse()
        for line in tmp2:
            print(line)
        self.blocks[x][y] = tmp


class BlockWorldState(MDPStateClass):
    def __init__(self, x, y, is_terminal=False):
        """
        A state in MDP
        :param x: <int>
        :param y: <int>
        :param is_terminal: <bool>
        """
        self.x = x
        self.y = y
        super().__init__(data=(self.x, self.y), is_terminal=is_terminal)

    def __str__(self):
        return "s: ({0}, {1})".format(self.x, self.y)

    def __repr__(self):
        return self.__str__()


if __name__ == "__main__":
    ###########################
    # GET THE GRIDWORLD
    ###########################
    env = BlockWorld(
        name="blockworld",
        width=5,
        height=5,
        init_loc=(0, 0),
        goal_loc=(4, 4),
        walls_loc=((3, 1), (3, 2), (3, 3), (0, 2), (1, 2), (1, 1),),
        holes_loc=((2, 1), (3, 4)))
    env.set_slip_prob(0.0)
    env.set_step_cost(1.0)
    env.set_hole_cost(50.0)
    env.set_goal_reward(50.0)
    observation = env.reset()
    env.print_blockworld()
    for t in range(10):
        random_action = random.choice(ACTIONS)
        print(observation, env.get_state_count(observation), random_action)
        observation, reward, done, info = env.step(random_action)
        if done:
            print("The agent arrived at tearminal state.")
            print(env.get_params())
            print("Exit")
            exit()
        env.print_blockworld()
