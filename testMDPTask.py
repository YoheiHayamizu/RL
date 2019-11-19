from RL.mdp.MDPGridWorld import MDPGridWorld
import RL.mdp.GridWorldConstants as GridConstant
from RL.mdp.MDPGraphWorld import MDPGraphWorld
import RL.mdp.GraphWorldConstants as GraphConstant
from RL.mdp.MDPGraphWorld_hard import MDPGraphWorld as MDPGraphWorld_hard
import RL.mdp.GraphWorldConstants_hard as GraphConstant_hard

from RL.agent.qlearning import QLearningAgent
from RL.agent.sarsa import SarsaAgent
from RL.agent.rmax import RMAXAgent
from RL.testConstants import *

import time
import numpy as np
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt
import dill
import seaborn as sns;
import copy

sns.set()


def run_episodes(mdp, method, step=50, episode=100, s=0):
    timestep_list = list()
    cumulative_reward_list = list()
    for e in range(episode):
        print("-------- new episode: {0:04} starts --------".format(e))
        mdp.reset()
        cumulative_reward = 0.0
        method.reset_of_episode()
        state = mdp.get_cur_state()
        action = method.act(state)
        method.update(state, action, 0.0, learning=False)
        # record agent's log every 250 episode
        if e % 250 == 0 or e == 1:
            with open(PKL_DIR + "mdp_{0}_{1}_{2}_{3}.pkl".format(method.name, mdp.name, s, e), "wb") as f:
                dill.dump(mdp, f)
            method.q_to_csv(CSV_DIR + "qtable_{0}_{1}_{2}_{3}.csv".format(method.name, mdp.name, s, e))
            agent_to_pickle(method, PKL_DIR + "{0}_{1}_{2}_{3}.pkl".format(method.name, mdp.name, s, e))
        for t in range(1, step):
            # print(state.get_state(), action, end='')
            mdp, reward, done, info = mdp.step(action)
            cumulative_reward += reward
            # print(reward)
            state = mdp.get_cur_state()
            action = method.act(state)
            method.update(state, action, reward)
            if done:
                break
        timestep_list.append(method.step_number)
        cumulative_reward_list.append(cumulative_reward)

    return timestep_list, cumulative_reward_list


def run_experiment(mdp, methods, step=50, episode=100, seed=10):
    timestep_list_dict = defaultdict(lambda: defaultdict(list))
    cumulative_reward_dict = defaultdict(lambda: defaultdict(list))
    for method in methods:
        # Record how long each agent spends learning.
        print("Running experiment: \n" + str(method))

        # save timestep of each methods
        df_timestep = pd.DataFrame()
        df_cumulative = pd.DataFrame()
        for s in range(0, seed):
            tmp_mdp = copy.deepcopy(mdp)
            method.reset()
            print("-------- new seed: {0:02} starts --------".format(s))
            timestep_list, cumulative_reward_list = run_episodes(tmp_mdp, method, step, episode, s)
            timestep_list_dict[str(method)][s] = timestep_list
            cumulative_reward_dict[str(method)][s] = cumulative_reward_list

            # save mdp of last seed of run for each methods
            with open(PKL_DIR + "mdp_{0}_{1}_{2}.pkl".format(method.name, tmp_mdp.name, s), "wb") as f:
                dill.dump(tmp_mdp, f)
            method.q_to_csv(CSV_DIR + "qtable_{0}_{1}_{2}.csv".format(method.name, tmp_mdp.name, s))
            agent_to_pickle(method, PKL_DIR + "{0}_{1}_{2}.pkl".format(method.name, tmp_mdp.name, s))

            tmp_timestep = pd.DataFrame(timestep_list_dict[str(method)])
            df_timestep = episode_data_to_df(tmp_timestep, df_timestep, method, s,
                                             columns=("Timestep", "episode", "seed", "method"))
            tmp_timestep.to_csv(CSV_DIR + "timestep_{0}_{1}_{2}.csv".format(method.name, tmp_mdp.name, s))

            tmp_cumulative = pd.DataFrame(cumulative_reward_dict[str(method)])
            df_cumulative = episode_data_to_df(tmp_cumulative, df_timestep, method, s,
                                               columns=("Cumulative_Reward", "episode", "seed", "method"))
            tmp_cumulative.to_csv(CSV_DIR + "cumulative_reward_{0}_{1}_{2}.csv".format(method.name, tmp_mdp.name, s))

        df_timestep.to_csv(CSV_DIR + "timesteps_{0}_{1}_all.csv".format(method.name, mdp.name))
        df_cumulative.to_csv(CSV_DIR + "cumulative_rewards_{0}_{1}_all.csv".format(method.name, mdp.name))

    # # plot and save timestep
    # step_plot = pd.DataFrame(timestep_list_dict)
    # step_plot_df = pd.DataFrame()
    # for m in step_plot.keys():
    #     for s in step_plot[m].keys():
    #         step_plot_df = episode_data_to_df(step_plot[m], step_plot_df, m, s,
    #                                           columns=("Timestep", "episode", "seed", "method"))
    # save_figure(step_plot_df, FIG_DIR + "timesteps_{0}.png".format(mdp.name), loc='upper right', pos=(1, 1),
    #             columns=("Timestep", "episode", "seed", "method"))
    #
    # reward_plot = pd.DataFrame(cumulative_reward_dict)
    # reward_plot_df = pd.DataFrame()
    # for m in step_plot.keys():
    #     for s in step_plot[m].keys():
    #         reward_plot_df = episode_data_to_df(reward_plot[m], reward_plot_df, m, s,
    #                                             columns=("Cumulative_Reward", "episode", "seed", "method"))
    # save_figure(reward_plot_df, FIG_DIR + "cumulative_rewards_{0}.png".format(mdp.name), loc="lower right", pos=(1, 0),
    #             columns=("Cumulative_Reward", "episode", "seed", "method"))


def episode_data_to_df(tmp, df, method, seed, columns=("Timestep", "episode", "seed", "method")):
    plot_df = pd.DataFrame(columns=columns)
    plot_df.loc[:, columns[0]] = tmp[seed]
    plot_df.loc[:, columns[1]] = range(len(tmp[seed]))
    plot_df.loc[:, columns[2]] = seed
    plot_df.loc[:, columns[3]] = str(method)
    df = df.append(plot_df)
    return df


def agent_to_pickle(method, filename=None):
    if filename is None:
        filename = "{0}.pkl".format(method.name)
    with open(filename, "wb") as f:
        dill.dump(method, f)


def load_agent(filename):
    with open(filename, "rb") as f:
        return dill.load(f)


if __name__ == "__main__":
    np.random.seed(0)
    grid_world = MDPGridWorld(5, 5,
                              init_loc=(0, 0), goals_loc=[(4, 4)], walls_loc=[], holes_loc=[],
                              is_goal_terminal=True, is_rand_init=False,
                              slip_prob=0.5, step_cost=1.0, hole_cost=1.0,
                              name="gridworld_test")
    # graph_world = MDPGraphWorld(node_num=15, is_goal_terminal=True, step_cost=1.0,
    #                             success_rate=GraphConstant.success_rate_dict2)

    mdp = grid_world
    # mdp = graph_world_normal
    #
    qLearning = QLearningAgent(mdp.get_actions(), name="QLearning", alpha=0.1, gamma=0.99, epsilon=0.1,
                               explore="uniform")
    # sarsa = SarsaAgent(mdp.get_actions(), name="Sarsa", alpha=0.1, gamma=0.99, epsilon=0.1, explore="uniform")
    rmax = RMAXAgent(mdp.get_actions(), "RMAX", rmax=10, u_count=2, gamma=0.9, epsilon_one=0.99)

    # methods = [qLearning, sarsa, rmax]
    # methods = [qLearning, sarsa]
    methods = [qLearning]

    run_experiment(mdp, methods, step=100, seed=10, episode=300)
