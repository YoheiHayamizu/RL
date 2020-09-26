import pandas as pd
import numpy as np
import optparse

from exe.config import *


def run_episodes(env, agent, step=50, episode=100, s=0, display=False):
    data_list = list()
    for e in range(episode):
        # INIT ENV AND AGENT
        state = env.reset()
        agent.reset_of_episode()
        cumulative_reward = 0.0
        print("-------- new episode: {0:02} starts --------".format(e))
        for t in range(0, step):
            # agent update actions which can be selected at this step
            agent.set_actions(env.get_executable_actions(state))
            # agent selects an action
            action = agent.act(state)

            # EXECUTE ACTION
            next_state, reward, done, info = env.step(action)
            # agent updates values
            agent.update(state, action, reward, next_state, done)
            # update the cumulative reward
            cumulative_reward += reward

            if display:
                env.render()

            # END IF DONE
            if done:
                break

            # update the current state
            state = next_state

        #############
        # Logging
        ############
        data_list.append([e, agent.number_of_steps, cumulative_reward, s] +
                         list(env.get_params().values()) +
                         list(agent.get_params().values()))

    df = pd.DataFrame(data_list, columns=['Episode', 'Timestep', 'Cumulative Reward', 'seed'] +
                                         list(env.get_params().keys()) +
                                         list(agent.get_params().keys()))
    df.to_csv(LOG_DIR + "{0}_{1}_{2:02}_fin.csv".format(agent.name, env.name, s))
    env.to_pickle(LOG_DIR + "mdp_{0}_{1}_{2:02}_fin.pkl".format(agent.name, env.name, s))
    agent.to_pickle(LOG_DIR + "agent_{0}_{1}_{2:02}_fin.pkl".format(agent.name, env.name, s))


def runs_episodes(_mdp, _agent, step=50, episode=100, seed=10):
    print("Running experiment: {0} in {1}".format(_agent.name, _mdp.name))
    for s in range(0, seed):
        np.random.seed(s)
        _agent.reset()
        print("-------- new seed: {0:02} starts --------".format(s))
        run_episodes(_mdp, _agent, step, episode, s)


def parse_options():
    usage = "usage: %prog [options]"
    parser = optparse.OptionParser(usage=usage)
    parser.add_option('--gamma',
                      action='store', type='float', dest='gamma', default=0.95,
                      help='Discount on future (default %default)')
    parser.add_option('--epsilon',
                      action='store', type='float', dest='epsilon', default=0.1, metavar="EPSILON",
                      help='Chance of taking a random action in q-learning (default %default)')
    parser.add_option('--alpha',
                      action='store', type='float', dest='alpha', default=0.1, metavar="ALPHA",
                      help='TD learning rate (default %default)')
    parser.add_option('-i', '--iterations',
                      action='store', type='int', dest='iters', default=100, metavar="STEP",
                      help='Number of rounds of value iteration (default %default)')
    parser.add_option('-e', '--episodes',
                      action='store', type='int', dest='episodes', default=1000, metavar="EPISODE",
                      help='Number of epsiodes of the MDP to run (default %default)')
    parser.add_option('-s', '--seed',
                      action='store', type='int', dest='seeds', default=1, metavar="SEED",
                      help='Number of seeds of the experiment to run (default %default)')
    parser.add_option('-n', '--lookahead', action='store', type='int', dest='lookahead', default=10, metavar="N",
                      help='Number of times of the planning to look ahead (default %default)')
    parser.add_option('-r', '--rmax',
                      action='store', type='float', dest='rmax', default=1000,
                      help='The upper bound of the reward function (default %default)')
    parser.add_option('-u', '--update_count',
                      action='store',
                      type='int', dest='u_count', default=2, metavar="U_COUNT",
                      help='Number of count of updating q-values (default %default)')
    parser.add_option('-a', '--agent',
                      action='store', metavar="AGENT", type='string', dest='agent', default="random",
                      help="""Agent type (options are  default %default)""")
    parser.add_option('--mdpName',
                      action='store', metavar="MDP_NAME", type='string', dest='mdpName', default="testGraph",
                      help="""MDP name (options are default %default)""")
    parser.add_option('--mappath',
                      action='store', metavar="path", type='string', dest='path', default="../mdp/graphworld/map.json",
                      help="""Map Path (options are default %default)""")

    opts, args = parser.parse_args()

    return opts
