import numpy as np
import matplotlib.pyplot as plt
import gym


# 一回の試行を回す関数
def do_episode(w, env):
    done = False
    observation = env.reset()
    num_steps = 0

    while not done and num_steps <= max_number_of_steps:
        action = take_action(observation, w)
        observation, _, done, _ = env.step(action)
        num_steps += 1

    step_val = -1 if num_steps >= max_number_of_steps else num_steps - max_number_of_steps
    return step_val, num_steps


def take_action(X, w):  # 値が0を超えたら1を返すようにする
    action = 1 if calculate(X, w) > 0.0 else 0
    return action


def calculate(X, w):
    result = np.dot(X, w)  # 返り値は配列ではなく，1つの値にする
    return result


def make_params(shape_list):
    weight_list = []
    for i in range(len(shape_list) - 1):
        weight = np.random.randn(shape_list[i], shape_list[i + 1])
        weight_list.append(weight)
    return weight_list


env = gym.make('CartPole-v0')


eta = 0.2
sigma = 0.05

max_episodes = 5000
max_number_of_steps = 200
n_states = 4
num_batch = 10
num_consecutive_iterations = 100

w = np.random.randn(n_states)
reward_list = np.zeros(num_batch)
reward_h = []
last_time_steps = np.zeros(num_consecutive_iterations)
mean_list = []

for episode in range(max_episodes // num_batch):
    N = np.random.normal(scale=sigma, size=(num_batch, w.shape[0]))  # パラメータの値を変動させるための値．これが偏差となる．

    for j in range(num_batch):
        w_try = w + N[j]
        reward, steps = do_episode(w_try, env)
        if j == num_batch - 1:
            print('%d Episode finished after %d steps / mean %f' % (episode * num_batch, steps, last_time_steps.mean()))
        last_time_steps = np.hstack((last_time_steps[1:], [steps]))
        reward_list[j] = reward
        mean_list.append(last_time_steps.mean())
        if last_time_steps.mean() >= 195:
            break

        std = np.std(reward_list)
        if std == 0:
            std = 1

        A = (reward_list - np.mean(reward_list)) / std
        w_delta = eta / (num_batch * sigma) * np.dot(N.T, A)

        w += w_delta
    env.render()

env.close()

plt.plot(mean_list)
plt.show()
