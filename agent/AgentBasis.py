import pandas as pd
import dill


class AgentBasisClass:
    def __init__(self, name, actions=None, gamma=0.99):
        self.name = name
        self.actions = actions
        self.gamma = gamma
        self.number_of_episode = 0
        self.step_number = 0

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

    # Setter

    def set_name(self, new_name):
        self.name = new_name

    def set_gamma(self, new_gamma):
        self.gamma = new_gamma

    def set_actions(self, new_actions):
        self.actions = new_actions

    # Core

    def act(self, state): ...

    def reset(self):
        self.number_of_episode = 0

    def reset_of_episode(self):
        self.step_number = 0

    def q_to_csv(self, filename):
        table = pd.DataFrame(self.Q, dtype=str)
        table.to_csv(filename)

    def to_pickle(self, filename):
        with open(filename, "wb") as f:
            dill.dump(self, f)
