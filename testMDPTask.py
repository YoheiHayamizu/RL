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
import optparse

sns.set()


def run_episodes(_mdp, _agent, step=50, episode=100, s=0):
    df_list = list()
    for e in range(episode):
        print("-------- new episode: {0:04} starts --------".format(e))
        _mdp.reset()
        _agent.reset_of_episode()
        cumulative_reward = 0.0
        state = _mdp.get_cur_state()
        action = _agent.act(state)
        _agent.update(state, action, 0.0, learning=False)
        env.print_gird()
        for t in range(1, step):
            _mdp, reward, done, info = _mdp.step(action)
            cumulative_reward += reward
            print(state.get_state(), action, reward, _mdp.get_visited(state))
            if done:
                print("The agent arrived at tearminal state.")
                print("Exit")
                break
            state = _mdp.get_cur_state()
            action = _agent.act(state)
            _agent.update(state, action, reward)
            env.print_gird()

        #############
        # Logging
        #############
        if e % int(episode/2) == 0:  # record agent's log every 250 episode
            _mdp.to_pickle(PKL_DIR + "mdp_{0}_{1}_{2}_{3}.pkl".format(_agent.name, _mdp.name, s, e))
            _agent.to_pickle(PKL_DIR + "{0}_{1}_{2}_{3}.pkl".format(_agent.name, _mdp.name, s, e))
            _agent.q_to_csv(CSV_DIR + "q_{0}_{1}_{2}_{3}.csv".format(_agent.name, _mdp.name, s, e))
        df_list.append([e, _agent.step_number, cumulative_reward, s, _agent.name])
    df = pd.DataFrame(df_list, columns=['Episode', 'Timestep', 'Cumulative Reward', 'seed', 'Name'])
    df.to_csv(CSV_DIR + "{0}_{1}_{2}_fin.csv".format(_agent.name, _mdp.name, s))
    _mdp.to_pickle(PKL_DIR + "mdp_{0}_{1}_{2}_fin.pkl".format(_agent.name, _mdp.name, s))
    _agent.q_to_csv(CSV_DIR + "q_{0}_{1}_{2}_fin.csv".format(_agent.name, _mdp.name, s))
    _agent.to_pickle(PKL_DIR + "{0}_{1}_{2}_fin.pkl".format(_agent.name, _mdp.name, s))


def runs_episodes(_mdp, _agent, step=50, episode=100, seed=10):
    print("Running experiment: {0}".format(str(_agent)))
    for s in range(0, seed):
        _agent.reset()
        print("-------- new seed: {0:02} starts --------".format(s))
        run_episodes(_mdp, _agent, step, episode, s)


def run_experiments(_mdp, _agents, step=50, episode=100, seed=10):
    for a in _agents:
        runs_episodes(_mdp, a, step, episode, seed)


def episode_data_to_df(tmp, df, agent, seed, columns=("Timestep", "episode", "seed", "agent")):
    plot_df = pd.DataFrame(columns=columns)
    plot_df.loc[:, columns[0]] = tmp[seed]
    plot_df.loc[:, columns[1]] = range(len(tmp[seed]))
    plot_df.loc[:, columns[2]] = seed
    plot_df.loc[:, columns[3]] = str(agent)
    df = df.append(plot_df)
    return df


def load_agent(filename):
    with open(filename, "rb") as f:
        return dill.load(f)


def parseOptions():
    optParser = optparse.OptionParser()
    optParser.add_option('-d', '--discount', action='store',
                         type='float', dest='discount', default=0.9,
                         help='Discount on future (default %default)')
    optParser.add_option('-e', '--epsilon', action='store',
                         type='float', dest='epsilon', default=0.1,
                         metavar="E", help='Chance of taking a random action in q-learning (default %default)')
    optParser.add_option('-l', '--learningRate', action='store',
                         type='float', dest='learningRate', default=0.5,
                         metavar="P", help='TD learning rate (default %default)')
    optParser.add_option('-i', '--iterations', action='store',
                         type='int', dest='iters', default=10,
                         metavar="K", help='Number of rounds of value iteration (default %default)')
    optParser.add_option('-k', '--episodes', action='store',
                         type='int', dest='episodes', default=1,
                         metavar="K", help='Number of epsiodes of the MDP to run (default %default)')
    optParser.add_option('-g', '--grid', action='store',
                         metavar="G", type='string', dest='grid', default="BookGrid",
                         help='Grid to use (case sensitive; options are BookGrid, BridgeGrid, CliffGrid, '
                              'MazeGrid, default %default)')
    optParser.add_option('-a', '--agent', action='store', metavar="A",
                         type='string', dest='agent', default="random",
                         help='Agent type (options are \'random\', \'value\' and \'q\', default %default)')
    # optParser.add_option('-r', '--livingReward', action='store',
    #                      type='float', dest='livingReward', default=0.0,
    #                      metavar="R", help='Reward for living for a time step (default %default)')
    # optParser.add_option('-n', '--noise', action='store',
    #                      type='float', dest='noise', default=0.2,
    #                      metavar="P", help='How often action results in ' +
    #                                        'unintended direction (default %default)')
    # optParser.add_option('-w', '--windowSize', metavar="X", type='int', dest='gridSize', default=150,
    #                      help='Request a window width of X pixels *per grid cell* (default %default)')
    # optParser.add_option('-t', '--text', action='store_true',
    #                      dest='textDisplay', default=False,
    #                      help='Use text-only ASCII display')
    # optParser.add_option('-p', '--pause', action='store_true',
    #                      dest='pause', default=False,
    #                      help='Pause GUI after each time step when running the MDP')
    # optParser.add_option('-q', '--quiet', action='store_true',
    #                      dest='quiet', default=False,
    #                      help='Skip display of any learning episodes')
    # optParser.add_option('-s', '--speed', action='store', metavar="S", type=float,
    #                      dest='speed', default=1.0,
    #                      help='Speed of animation, S > 1.0 is faster, 0.0 < S < 1.0 is slower (default %default)')
    # optParser.add_option('-v', '--valueSteps', action='store_true', default=False,
    #                      help='Display each step of value iteration')
    optParser.add_option('-m', '--manual', action='store_true',
                         dest='manual', default=False,
                         help='Manually control agent')

    opts, args = optParser.parse_args()

    if opts.manual and opts.agent != 'q':
        print('## Disabling Agents in Manual Mode (-m) ##')
        opts.agent = None

    # MANAGE CONFLICTS
    if opts.textDisplay or opts.quiet:
        # if opts.quiet:
        opts.pause = False
        # opts.manual = False

    if opts.manual:
        opts.pause = True

    return opts


if __name__ == "__main__":
    np.random.seed(0)
    ###########################
    # GET THE GRIDWORLD
    ###########################
    width, height = 5, 5
    init_loc = (0, 0)
    goals_loc = ((4, 4), )
    walls_loc = ()
    holes_loc = ()
    env = MDPGridWorld(width, height, init_loc, goals_loc, walls_loc, holes_loc, name="testEnv")
    env.set_slip_prob(0.0)
    env.set_step_cost(0.0)
    env.set_hole_cost(1.0)
    env.set_goal_reward(1)
    mdp = env

    ###########################
    # GET THE AGENT
    ###########################
    qLearning = QLearningAgent(mdp.get_actions(), name="QLearning", alpha=0.1, gamma=0.99, epsilon=0.1,
                               explore="uniform")
    # sarsa = SarsaAgent(mdp.get_actions(), name="Sarsa", alpha=0.1, gamma=0.99, epsilon=0.1, explore="uniform")
    # rmax = RMAXAgent(mdp.get_actions(), "RMAX", rmax=10, u_count=2, gamma=0.9, epsilon_one=0.99)
    agent = qLearning
    # agents = [qLearning, sarsa, rmax]

    ###########################
    # RUN
    ###########################
    run_episodes(mdp, agent, step=50, episode=100)
    # run_experiments(mdp, agents, step=50, episode=100, seed=5)

    # ###########################
    # # GET THE DISPLAY ADAPTER
    # ###########################
    # import RL.mdp.display as graphic
    #
    # manual = False
    # quiet = False
    # display = graphic.GraphicsGridworldDisplay(env)
    # try:
    #     display.start()
    # except KeyboardInterrupt:
    #     sys.exit(0)
    #
    # display_callback = lambda x: None
    # if manual:
    #     display_callback = lambda state: display.displayNullValues(state)
    # else:
    #     display_callback = lambda state: display.displayQValues(a, state, "CURRENT Q-VALUES")
    #
    # message_callback = lambda x: print(x)
    # if quiet:
    #     message_callback = lambda x: None
    #
    # # FIGURE OUT WHETHER THE USER WANTS MANUAL CONTROL (FOR DEBUGGING AND DEMOS)
    # if manual:
    #     decisionCallback = lambda state: get_user_action(state, env.get_actions(state))
    # else:
    #     decisionCallback = a.getAction
