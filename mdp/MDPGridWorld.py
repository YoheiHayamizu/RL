from mdp.MDPBasis import MDPBasisClass
from mdp.MDPState import MDPStateClass
from mdp.GridWorldConstants import *
import random


class MDPGridWorld(MDPBasisClass):
    def __init__(self,
                 width=5, height=5,
                 init_loc=(0, 0), goals_loc=[(4, 4)], walls_loc=[(3, 3)], holes_loc=[],
                 is_goal_terminal=True, is_rand_init=False,
                 slip_prob=0.0, step_cost=0.0, hole_cost=1.0,
                 name="gridworld"
                 ):
        """
        Constructor of MDPMaze
        :param width: <int> width of maze
        :param height: <int> height of maze
        :param init_loc: <tuple> initial state
        :param goals_loc: <list<tuple>> goal states
        :param walls_loc: <list<tuple>> walls
        :param holes_loc: <list<tuple>> holes
        :param is_goal_terminal: <bool> if true, episode will terminate when agent reaches fin_locs
        :param is_rand_init: <bool> if true, init_loc is decided randomly
        :param slip_prob: <float> probability of that agent actions fail
        :param step_cost: <float> cost of one action
        :param hole_cost: <float> cost of dropping a hole
        :param name: <str> name of maze
        """
        if is_rand_init:
            init_loc = random.randint(1, width), random.randint(1, height)
            while init_loc in walls_loc:
                init_loc = random.randint(1, width), random.randint(1, height)
        self.init_loc = init_loc
        self.init_state = MDPGridWorldState(init_loc[0], init_loc[1])
        self.cur_state = self.init_state
        self.actions = ACTIONS
        self.goals = goals_loc
        self.walls = walls_loc
        self.holes = holes_loc

        self.width = width
        self.height = height
        self.is_goal_terminal = is_goal_terminal
        self.is_rand_init = is_rand_init
        self.slip_prob = slip_prob
        self.step_cost = step_cost
        self.hole_cost = hole_cost
        self.name = name
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
        get_params["cur_state"] = self.cur_state
        get_params["goals"] = self.goals
        get_params["walls"] = self.walls
        get_params["holes"] = self.holes
        get_params["is_goal_terminal"] = self.is_goal_terminal
        get_params["is_rand_init"] = self.is_rand_init
        get_params["slip_prob"] = self.slip_prob
        get_params["hole_cost"] = self.hole_cost
        return get_params

    def get_slip_prob(self):
        return self.slip_prob

    def get_hole_cost(self):
        return self.hole_cost

    def get_goals_loc(self):
        return self.goals

    def get_holes_loc(self):
        return self.holes

    # Setter

    def set_slip_prob(self, new_slip_prob):
        self.slip_prob = new_slip_prob

    def set_hole_cost(self, new_hole_cost):
        self.hole_cost = new_hole_cost

    def add_goals_loc(self, new_goal_loc):
        self.goals.append(new_goal_loc)

    def add_holes_loc(self, new_hole_loc):
        self.holes.append(new_hole_loc)

    # Core

    def _transition_func(self, state, action):
        """
        transition function. it returns next state
        :param state: <State>
        :param action: <str>
        :return: next_state <State>
        """

        if state._is_terminal:
            return state

        if self.slip_prob > random.random():
            action = random.choice(ACTIONS)

        if action not in ACTIONS:
            print("the action is not defined in transition function")
            next_state = MDPGridWorldState(state.x, state.y)
        elif action == "up" and state.y < self.height - 1 and not self.is_wall(state.x, state.y + 1):
            next_state = MDPGridWorldState(state.x, state.y + 1)
        elif action == "down" and state.y > 0 and not self.is_wall(state.x, state.y - 1):
            next_state = MDPGridWorldState(state.x, state.y - 1)
        elif action == "left" and state.x > 0 and not self.is_wall(state.x - 1, state.y):
            next_state = MDPGridWorldState(state.x - 1, state.y)
        elif action == "right" and state.x < self.width - 1 and not self.is_wall(state.x + 1, state.y):
            next_state = MDPGridWorldState(state.x + 1, state.y)
        else:
            next_state = MDPGridWorldState(state.x, state.y)

        if (next_state.x, next_state.y) in self.goals and self.is_goal_terminal:
            next_state.set_terminal(True)
        return next_state

    def is_wall(self, x, y):
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
        if next_state.is_terminal:
            return 1 - self.step_cost
        elif (next_state.x, next_state.y) in self.holes:
            return -self.hole_cost
        else:
            return 0 - self.step_cost


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

    def __str__(self):
        return "s: ({0}, {1})".format(self.x, self.y)

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        assert isinstance(other, MDPGridWorldState), "Arg object is not in" + type(self).__module__
        return self.x == other.x and self.y == other.y


if __name__ == "__main__":
    grid_world = MDPGridWorld(5, 5, goals_loc=[(3, 2), (4, 4)])
    grid_world.reset()
    observation = grid_world
    for t in range(100):
        action = random.choice(ACTIONS)
        print(observation.cur_state, action)
        observation, reward, done, info = grid_world.step(action)
        if done:
            print("Goal!")
            exit()
