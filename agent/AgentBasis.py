import numpy as np


class AgentBasisClass:
    def __init__(self, name, actions, gamma=0.99):
        self.name = name
        self.actions = actions
        self.gamma = gamma
        self.episode_number = 0
        self.pre_state = None
        self.pre_action = None

    def __str__(self):
        return str(self.name)

    def __repr__(self):
        return self.__str__()

    # Accessors

    def get_params(self):
        """
        Return parameters of this class
        :return: <dict>
        """
        params = dict()
        params["name"] = self.name
        params["actions"] = self.actions
        params["gamma"] = self.gamma
        return params

    def get_name(self):
        return self.name

    def get_gamma(self):
        return self.gamma

    def get_pre_state(self):
        return self.pre_state

    def get_pre_action(self):
        return self.pre_action

    # Setter

    def set_name(self, new_name):
        self.name = new_name

    def set_gamma(self, new_gamma):
        self.gamma = new_gamma

    def set_actions(self, new_actions):
        self.actions = new_actions

    def set_pre_state(self, state):
        self.pre_state = state

    def set_pre_action(self, action):
        self.pre_action = action

    # Core

    def act(self, state, reward): ...

    def reset(self):
        self.episode_number = 0
        self.pre_state = None
        self.pre_action = None
