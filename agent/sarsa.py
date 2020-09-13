from RL.agent.AgentBasis import AgentBasisClass
from collections import defaultdict
import numpy as np
import random


class SarsaAgent(AgentBasisClass):
    def __init__(self,
                 name="SarsaAgent",
                 alpha=0.1,
                 gamma=0.99,
                 epsilon=0.1,
                 actions=None,
                 explore="uniform"):
        super().__init__(name, actions, gamma)
        self.alpha = self.init_alpha = alpha
        self.epsilon = self.init_epsilon = epsilon
        self.explore = explore

        self.Q = defaultdict(lambda: defaultdict(lambda: 0.0))

        # Accessors

    def get_params(self):
        params = super().get_params()
        params["alpha"] = self.alpha
        params["epsilon"] = self.epsilon
        params["explore"] = self.explore
        return params

    def get_alpha(self):
        return self.alpha

    def get_q_val(self, state, action):
        return self.Q[state][action]

    def get_policy(self, state):
        return self._get_max_q_key(state)

    def get_value(self, state):
        return self._get_max_q_val(state)

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
            action = random.choice(self.actions)
        else:
            action = self._epsilon_greedy_policy(state)  # default

        self.number_of_steps += 1

        return action

    def update(self, state, action, reward, next_state, done=False, **kwargs):
        next_action_value = 0
        if not done:
            if self.explore == "uniform": next_action = self._epsilon_greedy_policy(state)
            elif self.explore == "softmax": next_action = self._soft_max_policy(state)
            elif self.explore == "random": next_action = random.choice(self.actions)
            else: next_action = self._epsilon_greedy_policy(state)  # default
            next_action_value = self.get_q_val(next_state, next_action)
        diff = self.gamma * next_action_value - self.get_q_val(state, action)
        self.Q[state][action] += self.alpha * (reward + diff)

    def reset(self):
        self.alpha = self.init_alpha
        self.epsilon = self.init_epsilon
        self.Q = defaultdict(lambda: defaultdict(lambda: 0.0))

    def _get_max_q_key(self, state):
        return self._get_max_q(state)[0]

    def _get_max_q_val(self, state):
        return self._get_max_q(state)[1]

    def _get_max_q(self, state):
        tmp = self.actions
        best_action = random.choice(tmp)
        actions = tmp[:]
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
            action = random.choice(self.actions)
        else:
            action = self._get_max_q_key(state)
        return action
