import copy
import dill
from typing import Union, List, Any, Optional, Dict


class MDPBasisClass(object):
    """ abstract class for a MDP """
    def __init__(self,
                 init_state: Any,
                 transition_func: Any,
                 reward_func: Any,
                 actions: Any = None):
        self.init_state = copy.deepcopy(init_state)
        self.current_state = self.init_state
        self.actions = actions
        self.__transition_func = transition_func
        self.__reward_func = reward_func

    # Accessors

    def get_params(self) -> dict:
        """
        Returns:
            <dict> key -> param_name, val -> param_value
        """
        param_dict = dict()
        param_dict["init_state"] = self.init_state
        param_dict["actions"] = self.actions
        return param_dict

    def get_init_state(self):
        return self.init_state

    def get_current_state(self):
        return self.current_state

    def get_actions(self, state):
        return self.actions

    def get_transition_func(self):
        return self.__transition_func

    def get_reward_func(self):
        return self.__reward_func

    # Setters

    def set_init_state(self, new_init_state):
        self.init_state = copy.deepcopy(new_init_state)

    def set_actions(self, new_actions):
        self.actions = new_actions

    def set_transition_func(self, new_transition_func):
        self.__transition_func = new_transition_func

    # Core

    def step(self, action):
        """
        :param action: <str>
        :return: observation: <MDPStateClass>,
                 reward: <float>,
                 done: <bool>,
                 info: <dict>
        """
        next_state = self.__transition_func(self.current_state, action)
        reward = self.__reward_func(self.current_state, action, next_state)
        done = self.current_state.is_terminal()
        self.current_state = next_state

        return next_state, reward, done, self.get_params()

    def reset(self):
        self.current_state = self.init_state
        return self.current_state

    def to_pickle(self, filename):
        with open(filename, "wb") as f:
            dill.dump(self, f)
