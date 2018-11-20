import gym
import numpy as np
import matplotlib.pyplot as plt


def bins(clip_min, clip_max, num):
    return np.linspace(clip_min, clip_max, num + 1)[1:-1]


def digitize_state(_observation: object) -> object:
    cart_pos, cart_v, pole_angle, pole_v = _observation
    digitized = [np.digitize(cart_pos, bins=bins(-2.4, 2.4, 4)),
                 np.digitize(cart_v, bins=bins(-3.0, 3.0, 4)),
                 np.digitize(pole_angle, bins=bins(-0.5, 0.5, 4)),
                 np.digitize(pole_v, bins=bins(-2.0, 2.0, 4))]

    return sum([x * (4**i) for i, x in enumerate(digitized)])


class QLearning(object):
    def __init__(self, alpha=0.1, gamma=0.99):
        self.qValue = np.random.uniform(low=-1, high=1, size=(4**4, env.action_space.n))
        self.alpha = alpha
        self.gamma = gamma

    def update(self, _state, _action, _reward, _observation, _episode):
        next_state = digitize_state(_observation)
        next_act = self.get_action(next_state, _episode)
        self.qValue[_state, _action] = (1 - self.alpha) * self.qValue[_state, _action] + self.alpha * \
                                       (_reward + self.gamma * self.qValue[next_state, next_act])

        return next_state, next_act

    def get_action(self, _state, _episode):
        epsilon = 0.5 * (0.99 ** _episode)
        if epsilon <= np.random.uniform(0, 1):
            return np.argmax(self.qValue[_state])
        else:
            return np.random.choice([0, 1])


env = gym.make("CartPole-v0")

goal_average_steps = 195  # 195ステップ連続でポールが倒れないことを目指す
max_number_of_step = 200
num_consecutive_iterations = 100
num_episodes = 5000
last_time_steps = np.zeros(num_consecutive_iterations)
q_learning = QLearning()

step_list = []
for episode in range(num_episodes):
    observation = env.reset()

    state = digitize_state(observation)
    action = q_learning.get_action(state, episode)
    episode_reward = 0
    for t in range(max_number_of_step):
        # env.render()
        observation, reward, done, info = env.step(action)
        if done:
            reward -= 200
        episode_reward += reward
        state, action = q_learning.update(state, action, reward, observation, episode)
        if done:
            print('%d Episode finished after %f time steps / mean %f' % (episode, t + 1, last_time_steps.mean()))
            last_time_steps = np.hstack((last_time_steps[1:], [t + 1]))
            step_list.append(t + 1)
            break


print(q_learning.qValue)
plt.plot(np.arange(len(step_list)), step_list)
plt.xlabel('episode')
plt.ylabel('max_step')
env.close()






