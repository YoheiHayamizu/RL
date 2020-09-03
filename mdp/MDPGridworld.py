from mdp.MDPBasis import MDPBasisClass
from mdp.MDPState import MDPStateClass
import random
import copy

ROUND_OFF = 5
ACTIONS = ["up", "down", "left", "right"]

MAP = [""]


class MDPGridWorld(MDPBasisClass):
    def __init__(self, width=5, height=5, init_loc=(0, 0), goal_loc=(4, 4),
                 starts_loc=((0, 0),), goals_loc=((4, 4),), walls_loc=(), holes_loc=(), doors_loc=(),
                 is_goal_terminal=True, is_rand_init=False, is_rand_goal=False, grid_string=None,
                 slip_prob=0.0, step_cost=0.0, hole_cost=1.0, goal_reward=1, name="gridworld"
                 ):
        """
        Constructor of MDPMaze
        :param width: <int> width of maze
        :param height: <int> height of maze
        :param init_loc: <tuple> initial state
        :param goals_loc: <tuple<tuple>> goal states
        :param walls_loc: <tuple<tuple>> walls
        :param holes_loc: <tuple<tuple>> holes
        :param doors_loc: <tuple<tuple>> doors
        :param is_goal_terminal: <bool> if true, episode will terminate when agent reaches goals_loc
        :param is_rand_init: <bool> if true, init_loc is decided randomly
        :param slip_prob: <float> probability of that agent actions fail
        :param step_cost: <float> cost of one action
        :param hole_cost: <float> cost of dropping a hole
        :param name: <str> name of maze
        """

        self.name = name
        self.width = width
        self.height = height
        self.goals = goals_loc
        self.starts = starts_loc
        self.walls = walls_loc
        self.holes = holes_loc
        self.doors = doors_loc
        self.init_loc = init_loc
        self.goal_loc = goal_loc
        self.slip_prob = slip_prob
        self.step_cost = step_cost
        self.hole_cost = hole_cost
        self.goal_reward = goal_reward
        self.is_rand_init = is_rand_init
        self.is_rand_goal = is_rand_goal
        if grid_string is not None:
            self.grid = self.make_grid(grid_string)
        else:
            self.grid = self.conv_grid()
        self.init_grid = copy.deepcopy(self.grid)
        self.states = [[MDPGridWorldState(x, y) for y in range(len(self.grid[x]))] for x in range(len(self.grid))]
        self.init_state = self.states[self.init_loc[0]][self.init_loc[1]]
        self.goal_query = self.goal_loc
        self.set_rand_init()
        self.set_rand_goal()
        self.cur_state = self.init_state
        self.is_goal_terminal = is_goal_terminal
        self.set_actions(ACTIONS)
        super().__init__(self.init_state, self.actions, self._transition_func, self._reward_func,
                         self.step_cost)

    def __str__(self):
        return self.name + "_h-" + str(self.height) + "_w-" + str(self.width)

    def __repr__(self):
        return self.__str__()

    # Accessors

    def get_params(self):
        get_params = super().get_params()
        get_params["width"] = self.width
        get_params["height"] = self.height
        get_params["init_state"] = self.init_state
        get_params["goal_query"] = self.goal_query
        get_params["cur_state"] = self.cur_state
        get_params["goals"] = self.goals
        get_params["walls"] = self.walls
        get_params["holes"] = self.holes
        get_params["doors"] = self.doors
        get_params["is_goal_terminal"] = self.is_goal_terminal
        get_params["is_rand_init"] = self.is_rand_init
        get_params["slip_prob"] = self.slip_prob
        get_params["hole_cost"] = self.hole_cost
        return get_params

    def get_slip_prob(self):
        return self.slip_prob

    def get_hole_cost(self):
        return self.hole_cost

    def get_goal_reward(self):
        return self.goal_reward

    def get_goals_loc(self):
        return self.goals

    def get_holes_loc(self):
        return self.holes

    def get_doors_loc(self):
        return self.doors

    def get_state(self, x, y):
        return self.states[x][y]

    def get_states(self):
        flatten = lambda l: [item for sublist in l for item in sublist]
        return flatten(self.states)

    def get_actions(self, state=None):
        if state is None:
            return self.actions
        return self.actions[state.get_state()]

    def get_visited(self, state):
        return self.grid[state.x][state.y][1]

    # Setter

    def set_slip_prob(self, new_slip_prob):
        self.slip_prob = new_slip_prob

    def set_hole_cost(self, new_hole_cost):
        self.hole_cost = new_hole_cost

    def set_goal_reward(self, new_goal_reward):
        self.goal_reward = new_goal_reward

    def set_init_loc(self, new_init_loc):
        self.init_loc = new_init_loc

    def set_goal_locs(self, new_goal_locs):
        self.goals = new_goal_locs

    def set_wall_locs(self, new_wall_locs):
        self.walls = new_wall_locs

    def set_door_locs(self, new_door_locs):
        self.doors = new_door_locs

    def set_hole_locs(self, new_hole_locs):
        self.holes = new_hole_locs

    def set_rand_init(self):
        if self.is_rand_init:
            self.init_loc = random.choice(self.starts)
        self.init_state = self.states[self.init_loc[0]][self.init_loc[1]]

    def set_rand_goal(self):
        if self.is_rand_goal:
            self.goal_loc = random.choice(self.goals)
        self.goal_query = self.goal_loc
        # print("New goal is {0}".format(str(self.goal_query)))
        # if self.is_rand_init:
        #     init_loc = random.randint(1, self.width), random.randint(1, self.height)
        #     while init_loc in self.walls + self.goals + self.holes + self.doors:
        #         init_loc = random.randint(1, self.width), random.randint(1, self.height)

    def add_goals_loc(self, new_goal_loc):
        self.goals.append(new_goal_loc)

    def add_holes_loc(self, new_hole_loc):
        self.holes.append(new_hole_loc)

    def set_actions(self, new_actions):
        self.actions = {"s: ({0}, {1})".format(i, j): ACTIONS for i in range(self.width) for j in range(self.height)}

    def visited(self, state):
        x, y = state.x, state.y
        self.grid[x][y][1] += 1

    # Core

    def reset(self):
        # print(self.cur_state)
        self.cur_state.set_terminal(False)
        self.set_rand_init()
        self.set_rand_goal()
        super().reset()

    def _transition_func(self, state, action):
        """
        transition function. it returns next state
        :param state: <State>
        :param action: <str>
        :return: next_state <State>
        """

        self.visited(state)

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
                next_state.x, next_state.y) == self.goal_loc) and self.is_goal_terminal:
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
        if (next_state.x, next_state.y) == self.goal_loc:
            return self.get_goal_reward()
        elif (next_state.x, next_state.y) in self.holes:
            return -self.get_hole_cost()
        else:
            return 0 - self.get_step_cost()

    def make_grid(self, grid_string):
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
        self.set_door_locs(tuple(new_door_locs))
        self.set_init_loc(new_init_loc)
        return grid

    def conv_grid(self):
        grid = [[[' ', 0] for y in range(self.height)] for x in range(self.width)]
        for el in self.goals:
            grid[el[0]][el[1]][0] = 'G'
        for el in self.holes:
            grid[el[0]][el[1]][0] = 'O'
        for el in self.walls:
            grid[el[0]][el[1]][0] = '#'
        x, y = self.init_loc
        grid[x][y][0] = 'S'
        return grid

    def print_gird(self):
        x, y = self.cur_state.x, self.cur_state.y
        tmp = self.grid[x][y][0]
        self.grid[x][y][0] += '*'
        t = [[self.grid[x][y][0] for x in range(self.width)] for y in range(self.height)]
        t.reverse()
        for line in t:
            print(line)
        self.grid[x][y][0] = tmp


def get_maze_grid():
    grid = [[' ', ' ', ' ', 'G'],
            ['#', '#', ' ', '#'],
            [' ', '#', ' ', ' '],
            [' ', '#', '#', ' '],
            ['S', ' ', ' ', ' ']]
    return grid


class MDPGridWorldState(MDPStateClass):
    def __init__(self, x, y, is_terminal=False):
        """
        A state in MDP
        :param x: <float>
        :param y: <float>
        :param is_terminal: <bool>
        """
        self.x = round(x, ROUND_OFF)
        self.y = round(y, ROUND_OFF)
        super().__init__(data=(self.x, self.y), is_terminal=is_terminal)

    def __hash__(self):
        return hash(tuple(self.data))

    def __str__(self):
        return "s: ({0}, {1})".format(self.x, self.y)

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        assert isinstance(other, MDPGridWorldState), "Arg object is not in" + type(self).__module__
        return self.x == other.x and self.y == other.y

    def get_state(self):
        return self.__str__()


if __name__ == "__main__":
    ###########################
    # GET THE GRIDWORLD
    ###########################
    width, height = 5, 5
    init_loc = (3, 0)
    starts_loc = ((4, 0),)
    goals_loc = ((2, 2),)
    walls_loc = ((3, 1), (3, 2), (3, 3), (0, 2), (1, 2), (1, 1),)
    holes_loc = ((2, 1),)
    env = MDPGridWorld(width, height, starts_loc=starts_loc,
                       is_rand_init=False, is_rand_goal=False,
                       init_loc=init_loc, goal_loc=goals_loc[0],
                       goals_loc=goals_loc,
                       walls_loc=walls_loc, holes_loc=holes_loc, name='test')
    env.set_slip_prob(0.2)
    env.set_step_cost(1.0)
    env.set_hole_cost(50.0)
    env.set_goal_reward(50.0)
    env.reset()
    observation = env
    print(observation.get_cur_state())
    env.print_gird()
    for t in range(10):
        random_action = random.choice(ACTIONS)
        print(observation.get_cur_state(), random_action)
        observation, reward, done, info = env.step(random_action)
        if done:
            print("The agent arrived at tearminal state.")
            print(observation.get_params())
            print("Exit")
            exit()
        env.print_gird()
