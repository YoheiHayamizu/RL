from mdp.MDPGridWorld import MDPGridWorld
import mdp.GridWorldConstants as GridConstant
from mdp.MDPGraphWorld import MDPGraphWorld
import mdp.GraphWorldConstants as GraphConstant

from agent.qlearning import QLearningAgent
from agent.sarsa import SarsaAgent
from agent.rmax import RMAXAgent
from testConstants import *

import time
import numpy as np
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns;

sns.set()


def run_experiment(mdp, methods, step=50, episode=100, seed=10):
    time_dict = defaultdict(float)
    step_list_dict = defaultdict(lambda: defaultdict(list))
    for method in methods:
        # Record how long each agent spends learning.
        print("Running experiment: \n" + str(method))

        start = time.perf_counter()

        for s in range(seed):
            method.reset()
            print("-------- new seed: {0:02} starts --------".format(s))
            reward_list = run_episodes(mdp, method, step, episode)
            step_list_dict[str(method)][s] = reward_list

        end = time.perf_counter()
        time_dict[str(method)] = round(end - start, GridConstant.ROUND_OFF)

    # plot and save step time
    step_plot = pd.DataFrame(step_list_dict)
    step_plot_df = pd.DataFrame()
    for m in step_plot.keys():
        for s in step_plot[m].keys():
            step_plot_df = step_plot_df.append(episode_data_to_df(step_plot, m, s))
    save_figure(step_plot_df, FIG_DIR + "converge_{0}.png".format(mdp.name))

    # plot and save run time
    time_plot = pd.DataFrame(time_dict.items(), columns=["method", "time"])
    fig, ax = plt.subplots()
    ax.bar(time_plot["method"], time_plot["time"], width=0.2)
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Run Time[s]')
    ax.set_title('Run Time by methods')
    ax.set_xticks(range(len(time_plot["method"])))
    plt.savefig(FIG_DIR + "runtime_{0}.png".format(mdp.name))

    # q table to csv
    for method in methods:
        method.q_to_csv(FIG_DIR + "qtable_{0}.csv".format(method.name))


def run_episodes(mdp, method, step=50, episode=100):
    step_list = list()
    for e in range(episode):
        print("-------- new episode: {0:04} starts --------".format(e))
        mdp.reset()
        method.reset_of_episode()
        state = mdp.get_cur_state()
        action = method.act(state)
        for t in range(1, step):
            method.set_actions(state, mdp.get_actions())
            # print(method.actions[state])
            mdp, reward, done, info = mdp.step(action)
            state = mdp.get_cur_state()
            action = method.act(state)
            method.update(state, action, reward)
            # print(method.Q[state], reward)
            # print(mdp.get_cur_state(), action)
            # print(method.Q)
            if done:
                break
        step_list.append(method.step_number)
    return step_list


def save_figure(df, filename="figure.png"):
    fig, ax = plt.subplots()
    sns.lineplot(x="episode", y="steps", hue="method", data=df, ax=ax)
    plt.title(filename)
    plt.legend(loc='upper right', bbox_to_anchor=(0, 0), ncol=1)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles[1:], labels=labels[1:], loc=4)
    plt.savefig(filename)
    del fig
    del ax


def episode_data_to_df(_dict, method, seed):
    plot_df = pd.DataFrame(columns={"steps", "episode", "seed", "method"})
    plot_df.loc[:, "steps"] = _dict[method][seed]
    plot_df.loc[:, "episode"] = range(len(_dict[method][seed]))
    plot_df.loc[:, "seed"] = seed
    plot_df.loc[:, "method"] = str(method)
    return plot_df


if __name__ == "__main__":
    np.random.seed(0)
    grid_world = MDPGridWorld(5, 5,
                              init_loc=(0, 0), goals_loc=[(4, 4)], walls_loc=[], holes_loc=[],
                              is_goal_terminal=True, is_rand_init=False,
                              slip_prob=0.0, step_cost=1.0, hole_cost=1.0,
                              name="gridworld")
    graph_world = MDPGraphWorld(node_num=15, is_goal_terminal=True, step_cost=1.0, success_rate=GraphConstant.success_rate_dict2)

    # mdp = grid_world
    mdp = graph_world

    qLearning = QLearningAgent(mdp.get_actions(), name="QLearning", alpha=0.1, gamma=0.99, epsilon=0.1, explore="uniform")
    sarsa = SarsaAgent(mdp.get_actions(), name="Sarsa", alpha=0.1, gamma=0.99, epsilon=0.1, explore="uniform")
    rmax = RMAXAgent(mdp.get_actions(), "RMAX", rmax=10.0, u_count=2, gamma=0.95, epsilon_one=0.99)

    methods = [qLearning, sarsa, rmax]
    # methods = [qLearning, sarsa]
    # methods = [rmax]

    run_experiment(mdp, methods, step=100, seed=10, episode=500)


