from mdp.MDPGridWorld import MDPGridWorld
from mdp.GridWorldConstants import *

from agent.qlearning import QLearningAgent
from agent.sarsa import SarsaAgent
from testConstants import *

import time
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns;

sns.set()

from collections import defaultdict


def run_experiment(mdp, methods, step=50, episode=100, seed=10):
    time_dict = defaultdict(float)
    step_list_dict = defaultdict(lambda: defaultdict(list))
    for method in methods:
        # Record how long each agent spends learning.
        print("Running experiment: \n" + str(method))

        start = time.perf_counter()

        for s in range(seed):
            method.reset()
            reward_list = run_episodes(mdp, method, step, episode)
            step_list_dict[str(method)][s] = reward_list

        end = time.perf_counter()
        time_dict[str(method)] = round(end - start, ROUND_OFF)

    tmp = pd.DataFrame(step_list_dict)
    tmp_plot_df = pd.DataFrame()
    for method in methods:
        for s in range(seed):
            tmp_plot_df = tmp_plot_df.append(episode_data_to_df(tmp, str(method), s, episode), ignore_index=True)
    save_figure(tmp_plot_df, FIG_DIR+"converge.png")
    # show_figure(tmp_plot_df)


def show_figure(df):
    fig, ax = plt.subplots()
    sns.lineplot(x="episode", y="steps", hue="method", data=df, ax=ax)
    plt.legend(loc='upper right', bbox_to_anchor=(1, 1), ncol=1)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles[1:], labels=labels[1:], loc=4)
    plt.show()


def save_figure(df, filename="figure.png"):
    fig, ax = plt.subplots()
    sns.lineplot(x="episode", y="steps", hue="method", data=df, ax=ax)
    plt.title(filename)
    plt.legend(loc='upper right', bbox_to_anchor=(1, 1), ncol=1)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles[1:], labels=labels[1:], loc=4)
    plt.savefig(filename)
    del fig
    del ax


def episode_data_to_df(_dict, method, seed, episode):
    plot_df = pd.DataFrame(columns={"steps", "episode", "seed", "method"})
    plot_df.loc[:, "steps"] = _dict[method][seed]
    plot_df.loc[:, "episode"] = range(episode)
    plot_df.loc[:, "seed"] = seed
    plot_df.loc[:, "method"] = str(method)
    return plot_df


def run_episodes(mdp, method, step=50, episode=100):
    step_list = list()
    for e in range(episode):
        mdp.reset()
        state = mdp.get_cur_state()
        action = method.act(state)
        for t in range(step):
            mdp, reward, done, info = mdp.step(action)
            state = mdp.get_cur_state()
            action = method.act(state)
            method.update(state, action, reward)
            if done:
                break
        step_list.append(method.step_number)
        method.end_of_episode()
    return step_list


if __name__ == "__main__":
    np.random.seed(0)
    grid_world = MDPGridWorld(5, 5,
                              init_loc=(0, 0), goals_loc=[(4, 4)], walls_loc=[], holes_loc=[],
                              is_goal_terminal=True, is_rand_init=False,
                              slip_prob=0.0, step_cost=0.0, hole_cost=1.0,
                              name="gridworld")

    qLearning = QLearningAgent(ACTIONS, name="QLearning", alpha=0.1, gamma=0.99, epsilon=0.1, explore="uniform")
    sarsa = SarsaAgent(ACTIONS, name="Sarsa", alpha=0.1, gamma=0.99, epsilon=0.1, explore="uniform")
    methods = [qLearning, sarsa]
    run_experiment(grid_world, methods, step=50, seed=10, episode=500)
