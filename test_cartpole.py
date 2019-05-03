import gym
import qlearning
import sarsa
import cv2
import matplotlib.pyplot as plt

EPISODE = 2000
TIMESTEP = 200
RECORD_EPS = [100, 500, 1000, 2000]
VIDEO = False  # record flag
MOVIE_DIR = "./movies/"
IMG_DIR = "./images/"
FIG_DIR = "./figures/"


def display_frames_as_gif(frames, i_episode, _dir):
    framelist = list()
    imgsize_x = frames[0].shape[1]
    imgsize_y = frames[0].shape[0]
    for i, frame in enumerate(frames):
        framename = IMG_DIR + _dir + 'cartpole_{0:04d}_{1:03d}.png'.format(i_episode + 1, i + 1)
        cv2.imwrite(framename, frame)
        framelist.append(framename)

    # create movie
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    video = cv2.VideoWriter(MOVIE_DIR + _dir + 'cartpole_{0:04d}.mp4'.format(i_episode + 1),
                            fourcc, 20.0, (imgsize_x, imgsize_y))

    for img_file in framelist:
        img = cv2.imread(img_file)
        video.write(img)
    video.release()
    del fourcc
    del video



def discrete_qlearning():
    env = gym.make('CartPole-v0')
    agent = qlearning.QLearning(4, 2)
    timestep_list = list()

    for episode in range(EPISODE):
        observation = env.reset()
        action = env.action_space.sample()
        frames = list()
        for t in range(TIMESTEP):
            if episode + 1 in RECORD_EPS:
                frames.append(env.render(mode='rgb_array'))

            pre_observation = observation
            pre_action = action
            observation, reward, done, info = env.step(pre_action)
            action = agent.epsilon_greedy(observation, episode)
            if done:
                print("Episode finished after {} timesteps".format(t + 1))
                # average_step = average_step[1:] + [t]
                if t < TIMESTEP * 0.95:
                    reward = -0.1
                else:
                    reward = 1
                agent.update(pre_observation, pre_action, reward, observation)
                break
            else:
                reward = 0
                agent.update(pre_observation, pre_action, reward, observation)

        timestep_list.append(t)
        if episode + 1 in RECORD_EPS:
            display_frames_as_gif(frames, episode, "QLearning/")

    fig, ax = plt.subplots()
    ax.plot(range(len(timestep_list)), timestep_list)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Step")
    ax.set_title("QLearning")
    plt.savefig(FIG_DIR + "QLearning.png")
    env.close()
    return timestep_list


def discrete_sarsa():
    env = gym.make('CartPole-v0')
    agent = sarsa.Sarsa(4, 2)
    timestep_list = list()

    for episode in range(EPISODE):
        observation = env.reset()
        action = env.action_space.sample()
        frames = list()
        for t in range(TIMESTEP):
            if episode + 1 in RECORD_EPS and VIDEO:
                frames.append(env.render(mode='rgb_array'))

            pre_observation = observation
            pre_action = action
            observation, reward, done, info = env.step(pre_action)
            action = agent.get_max_action(observation)
            if done:
                print("Episode finished after {} timesteps".format(t + 1))
                # average_step = average_step[1:] + [t]
                if t < TIMESTEP * 0.95:
                    reward = -0.1
                else:
                    reward = 1
                agent.update(pre_observation, pre_action, reward, observation, action)
                break
            else:
                reward = 0
                agent.update(pre_observation, pre_action, reward, observation, action)

        timestep_list.append(t)
        if episode + 1 in RECORD_EPS and VIDEO:
            display_frames_as_gif(frames, episode, "Sarsa/")

    fig, ax = plt.subplots()
    ax.plot(range(len(timestep_list)), timestep_list)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Step")
    ax.set_title("Sarsa")
    plt.savefig(FIG_DIR + "Sarsa.png")
    env.close()
    return timestep_list


import csv
import pandas as pd
import seaborn as sns;

sns.set()

df = pd.DataFrame(columns={"steps", "episode", "seed", "method"})
for i in range(10):
    timestep_lists = discrete_qlearning()
    tmp = pd.DataFrame(columns={"steps", "episode", "seed"})
    tmp["steps"] = list(map(float, timestep_lists))
    tmp["episode"] = tmp.index
    tmp.loc[:, "seed"] = "{0}".format(i)
    df = pd.concat([df, tmp], ignore_index=True)

fig, ax = plt.subplots()
plotline = sns.lineplot(x="episode", y="steps", data=df, ax=ax)
plt.savefig(FIG_DIR + "QLearning-seed.png")

df = pd.DataFrame(columns={"steps", "episode", "seed"})
for i in range(10):
    timestep_lists = discrete_sarsa()
    tmp = pd.DataFrame(columns={"steps", "episode", "seed"})
    tmp["steps"] = list(map(float, timestep_lists))
    tmp["episode"] = tmp.index
    tmp.loc[:, "seed"] = "{0}".format(i)
    df = pd.concat([df, tmp], ignore_index=True)

fig, ax = plt.subplots()
plotline = sns.lineplot(x="episode", y="steps", data=df, ax=ax)
plt.savefig(FIG_DIR + "Sarsa-seed.png")