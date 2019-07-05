from agent.AgentBasis import AgentBasisClass
from collections import defaultdict
import numpy as np
import pandas as pd
import random


class RMAXAgent(AgentBasisClass):
    def __init__(self, actions, name="RMAXAgent", rmax=1.0, u_count=2, gamma=0.99, epsilon_one=0.1):
        super().__init__(name, actions, gamma)
        self.u_count, self.init_urate = u_count, u_count
        self.epsilon_one, self.init_epsilon_one = epsilon_one, epsilon_one
        self.rmax, self.init_rmax = rmax, rmax

        self.Q = defaultdict(lambda: defaultdict(lambda: self.rmax))
        self.V = defaultdict(lambda: 0.0)
        self.C_sa = defaultdict(lambda: defaultdict(lambda: 0))
        self.C_sas = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: 0)))
        self.rewards = defaultdict(lambda: defaultdict(list))

    # Accessors

    def get_params(self):
        params = self.get_params()
        params["urate"] = self.u_count
        params["Q"] = self.Q
        params["C_sa"] = self.C_sa
        params["C_sas"] = self.C_sas
        return params

    def get_urate(self):
        return self.u_count

    def get_q_val(self, state, action):
        return self.Q[state][action]

    def get_reward(self, state, action):
        if self.get_count(state, action) >= self.u_count:
            # print(float(sum(self.rewards[state][action])) / self.get_count(state, action))
            return float(sum(self.rewards[state][action])) / self.get_count(state, action)
        else:
            # print(self.rmax)
            return self.rmax

    def get_transition(self, state, action, next_state):
        # print(self.get_count(state, action, next_state) / self.get_count(state, action))
        return self.get_count(state, action, next_state) / self.get_count(state, action)

    def get_count(self, state, action, next_state=None):
        if next_state is None:
            return self.C_sa[state][action]
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

    def update(self, state, action, reward, learning=True):
        pre_state = self.get_pre_state()
        pre_action = self.get_pre_action()

        if learning:
            if pre_state is None and pre_action is None:
                self.set_pre_state(state)
                self.set_pre_action(action)
                return

            self.C_sa[pre_state][pre_action] += 1
            self.C_sas[pre_state][pre_action][state] += 1
            self.rewards[pre_state][pre_action] += [reward]
            if self.get_count(pre_state, pre_action) <= self.u_count:
                if self.u_count == self.get_count(pre_state, pre_action):
                    self._update_policy_iteration()

        self.set_pre_state(state)
        self.set_pre_action(action)

    def _update_policy_iteration(self):
        lim = int(np.log(1 / (self.epsilon_one * (1 - self.gamma))) / (1 - self.gamma))
        for l in range(1, lim):
            for s in self.C_sa.keys():
                for a in self.C_sa[s].keys():
                    if self.get_count(s, a) >= self.u_count:
                        self.Q[s][a] = self.get_reward(s, a) + self.gamma * \
                                       sum([(self.get_transition(s, a, sp) * self._get_max_q_val(sp))
                                            for sp in self.Q.keys()])

    # def _update_policy_iteration(self, tolerance=1e-6):
    #     dv = tolerance
    #     v = defaultdict(lambda: 0.0)
    #     while dv >= tolerance:
    #         dv = 0.0
    #         vi = v
    #         for s in self.C_sas.keys():
    #             for a in self.C_sas[s].keys():
    #                 for sp in self.C_sas[s][a].keys():
    #                     self.V[s] += self.get_transition(s, a, sp) * (self.get_reward(s, a, sp) + self.gamma * v[s])
    #             if abs(v[s] - vi[s]) > dv:
    #                 dv = abs(v[s] - vi[s])
    #     self.V = v
    #
    #     q = defaultdict(lambda: defaultdict(lambda: 0.0))
    #     for s in self.C_sas.keys():
    #         for a in self.C_sas[s].keys():
    #             for sp in self.C_sas[s][a].keys():
    #                 q[s][a] += self.get_transition(s, a, sp) * (self.get_reward(s, a, sp) + self.gamma * self.V[s])
    #     self.Q = q

    def reset(self):
        self.u_count = self.init_urate
        self.epsilon_one = self.init_epsilon_one
        self.episode_number = 0
        self.Q = defaultdict(lambda: defaultdict(lambda: self.rmax))
        self.C_sa = defaultdict(lambda: defaultdict(lambda: 0))
        self.C_sas = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: 0)))
        self.rewards = defaultdict(lambda: defaultdict(list))

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

    def q_to_csv(self, filename=None):
        if filename is None:
            filename = "qtable_{0}.csv".format(self.name)
        table = pd.DataFrame(self.Q)
        table.to_csv(filename)