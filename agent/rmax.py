from RL.agent.AgentBasis import AgentBasisClass
import numpy as np
import random
from collections import defaultdict
import itertools


class RMAXAgent(AgentBasisClass):
    def __init__(self,
                 name="RMAXAgent",
                 rmax=1.0,
                 u_count=2,
                 gamma=0.99,
                 actions=None,
                 epsilon=0.1):
        super().__init__(name, actions, gamma)
        self.u_count = self.init_urate = u_count
        self.epsilon = self.init_epsilon = epsilon
        self.rmax = self.init_rmax = rmax
        self.explore = "greedy"

        self.Q = defaultdict(lambda: defaultdict(lambda: self.rmax))
        self.V = defaultdict(lambda: 0.0)
        self.C_sas = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: 0)))
        self.rewards = defaultdict(lambda: defaultdict(list))

    # Accessors

    def get_params(self):
        params = self.get_params()
        params["urate"] = self.u_count
        params["Q"] = self.Q
        params["C_sas"] = self.C_sas
        return params

    def get_urate(self):
        return self.u_count

    def get_q_val(self, state, action):
        return self.Q[state][action]

    def get_policy(self, state):
        return self._get_max_q_key(state)

    def get_value(self, state):
        return self._get_max_q_val(state)

    def get_reward(self, state, action):
        if self.get_count(state, action) == 0:
            return 0.0
        return float(sum(self.rewards[state][action])) / self.get_count(state, action)

    def get_transition(self, state, action):
        next_state_probabilities = dict()
        for next_state in self.C_sas[state][action].keys():
            prob = self.get_count(state, action, next_state) / self.get_count(state, action)
            next_state_probabilities[next_state] = prob
        return next_state_probabilities

    def get_count(self, state, action, next_state=None):
        if next_state is None:
            return sum(self.C_sas[state][action].values())
        else:
            return self.C_sas[state][action][next_state]

    # Setters

    def set_urate(self, new_urate):
        self.u_count = new_urate

    # Core

    def act(self, state):
        action = self._get_max_q_key(state)

        self.step_number += 1

        return action

    def update(self, state, action, reward, next_state, done=False, **kwargs):
        if not done:
            self.C_sas[state][action][next_state] += 1
            self.rewards[state][action] += [reward]
        if self.get_count(state, action) == self.u_count:
            self._update_policy_iteration()
            # lim = int(np.log(1 / (self.epsilon * (1 - self.gamma))) / (1 - self.gamma))
            # for l in range(0, lim):
            #     for s in self.C_sas.keys():
            #         for a in self.C_sas[s].keys():
            #             if self.get_count(s, a) >= self.u_count:
            #                 next_state_probabilities = self.get_transition(s, a)
            #                 self.Q[s][a] = self.get_reward(s, a)
            #                 for ns, p in next_state_probabilities.items():
            #                     self.Q[s][a] += self.gamma * p * self._get_max_q_val(ns)
            #                     # print(s, a, self.Q[s][a])

    def _update_policy_iteration(self):
        lim = int(np.log(1 / (self.epsilon * (1 - self.gamma))) / (1 - self.gamma))
        tmp = list(map(lambda x: itertools.product(self.C_sas.keys(), self.C_sas[x].keys()), self.C_sas.keys())) * lim
        for l in range(0, lim):
            for s, a in tmp[l]:
                if self.get_count(s, a) >= self.u_count:
                    next_state_probabilities = self.get_transition(s, a)
                    self.Q[s][a] = self.get_reward(s, a) + self.gamma * sum(
                        [p * self._get_max_q_val(ns) for ns, p in next_state_probabilities.items()])
                    print(s, a, self.Q[s][a])

    def reset(self):
        self.u_count = self.init_urate
        self.epsilon = self.init_epsilon
        self.episode_number = 0
        self.Q = defaultdict(lambda: defaultdict(lambda: self.rmax))
        self.C_sas = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: 0)))
        self.rewards = defaultdict(lambda: defaultdict(list))

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
