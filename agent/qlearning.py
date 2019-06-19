from agent.AgentBasis import AgentBasisClass
from collections import defaultdict
import numpy as np


class QLearningAgent(AgentBasisClass):
    def __init__(self, actions, name="QLearningAgent", alpha=0.5, gamma=0.99, epsilon=0.1, explore="uniform"):
        super().__init__(name, actions, gamma)
        self.alpha = alpha
        self.epsilon = epsilon
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

    def act(self, state, reward, learning=True):
        if learning:
            self.update(self.pre_state, self.pre_action, reward, state)

        if self.explore == "uniform":
            action = self._epsilon_greedy_policy(state)
        elif self.explore == "softmax":
            action = self._soft_max_policy(state)
        elif self.explore == "random":
            action = np.random.choice(self.actions)
        else:
            action = self._epsilon_greedy_policy(state)  # default

        self.set_pre_state(state)
        self.set_pre_action(action)
        self.step_number += 1

        action = self._get_max_q_key(state)
        return action

    def update(self, state, action, reward, next_state):
        if state is None:
            return

        diff = self.gamma * self.Q[state][action] - self._get_max_q_val(next_state)
        self.Q[state][action] += self.alpha * (reward + diff)

    def _get_max_q_key(self, state):
        return self._get_max_q(state)[0]

    def _get_max_q_val(self, state):
        return self._get_max_q(state)[1]

    def _get_max_q(self, state):
        best_action = np.random.choice(self.actions)
        actions = self.actions[:]
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
        if self.epsilon > np.random.random():
            action = np.random.choice(self.actions)
        else:
            action = self._get_max_q_key(state)
        return action