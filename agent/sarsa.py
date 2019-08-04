from RL.agent.AgentBasis import AgentBasisClass
from collections import defaultdict
import numpy as np
import pandas as pd
import random


class SarsaAgent(AgentBasisClass):
    def __init__(self, actions, name="SarsaAgent", alpha=0.5, gamma=0.99, epsilon=0.1, explore="uniform"):
        super().__init__(name, actions, gamma)
        self.alpha, self.init_alpha = alpha, alpha
        self.epsilon, self.init_epsilon = epsilon, epsilon
        self.explore = explore
        self.step_number = 0

        self.Q = defaultdict(lambda: defaultdict(lambda: 0.0))

        # Accessors

    def get_params(self):
        params = self.get_params()
        params["alpha"] = self.alpha
        params["epsilon"] = self.epsilon
        params["explore"] = self.explore
        params["Q"] = self.Q
        return params

    def get_alpha(self):
        return self.alpha

    def get_q_val(self, state, action):
        return self.Q[state][action]

        # Setters

    def set_alpha(self, new_alpha):
        self.alpha = new_alpha

        # Core

    def act(self, state):
        if self.explore == "uniform":
            action = self._epsilon_greedy_policy(state)
        elif self.explore == "softmax":
            action = self._soft_max_policy(state)
        elif self.explore == "random":
            action = np.random.choice(self.actions[state])
        else:
            action = self._epsilon_greedy_policy(state)  # default

        self.step_number += 1

        return action

    def update(self, state, action, reward, learning=True):
        pre_state = self.get_pre_state()
        pre_action = self.get_pre_action()
        if learning:
            if pre_state is None:
                self.set_pre_state(state)
                self.set_pre_action(action)
                return

            diff = self.gamma * self.get_q_val(state, action) - self.get_q_val(pre_state, pre_action)
            self.Q[pre_state][pre_action] += self.alpha * (reward + diff)

        self.set_pre_state(state)
        self.set_pre_action(action)

    def reset(self):
        self.alpha = self.init_alpha
        self.epsilon = self.init_epsilon
        self.episode_number = 0
        self.Q = defaultdict(lambda: defaultdict(lambda: 0.0))

    def _get_max_q_key(self, state):
        return self._get_max_q(state)[0]

    def _get_max_q_val(self, state):
        return self._get_max_q(state)[1]

    def _get_max_q(self, state):
        best_action = random.choice(self.actions[state])
        actions = self.actions[state][:]
        np.random.shuffle(actions)
        max_q_val = float("-inf")
        for key in actions:
            q_val = self.get_q_val(state, key)
            if q_val > max_q_val:
                best_action = key
                max_q_val = q_val
        return best_action, max_q_val

    def _soft_max_policy(self, state):
        pass

    def _epsilon_greedy_policy(self, state):
        if self.epsilon > random.random():
            action = random.choice(self.actions[state])
        else:
            action = self._get_max_q_key(state)
        return action

    def q_to_csv(self, filename=None):
        if filename is None:
            filename = "qtable_{0}.csv".format(self.name)
        table = pd.DataFrame(self.Q, dtype=str)
        table.to_csv(filename)
