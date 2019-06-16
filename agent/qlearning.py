import numpy as np
import space


class QLearning:
    def __init__(self, dim_state, dim_action, alpha=0.5, gamma=0.99):
        self.num_state = space.NUM_DIGITIZED ** dim_state
        self.num_action = dim_action
        self.alpha = alpha
        self.gamma = gamma
        self.Q = np.zeros((self.num_state, self.num_action))

    def update(self, observation, action, reward, next_observation):
        state = space.digitize_state(observation)
        next_state = space.digitize_state(next_observation)
        next_action = self.get_max_action(next_observation)
        self.Q[state][action] += self.alpha * (
                reward + self.gamma * self.Q[next_state][next_action] - self.Q[state][action])

    def get_max_action(self, observation):
        state = space.digitize_state(observation)
        action, = np.where(self.Q[state] == np.max(self.Q[state]))
        return np.random.choice(action)

    def epsilon_greedy(self, observation, t):
        epsilon = np.exp(-t/10)
        if np.random.random() >= epsilon:
            action = self.get_max_action(observation)
        else:
            action = np.random.randint(0, 2)
        return action

    def greedy(self, observation):
        return self.get_max_action(observation)

