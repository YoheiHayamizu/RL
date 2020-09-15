import pandas as pd
import numpy as np

from exec.config import *
from mdp.graphworld.graphworld import GraphWorld
from agent.qlearning import QLearningAgent


def run_episodes(env, agent, step=50, episode=100, s=0, display_cb=None, display=False):
    if display_cb is None:
        display_cb = lambda state: env.render()
    data_list = list()
    for e in range(episode):
        # INIT ENV AND AGENT
        state = env.reset()
        agent.reset_of_episode()
        cumulative_reward = 0.0
        print("-------- new episode: {0:02} starts --------".format(e))
        for t in range(0, step):
            # agent selects an action
            action = agent.act(state)
            # print(state, action)
            # EXECUTE ACTION
            next_state, reward, done, info = env.step(action)
            # agent updates values
            agent.update(state, action, reward, next_state, done)
            # update the cumulative reward
            cumulative_reward += reward

            if display:
                display_cb(state)

            # END IF DONE
            if done:
                print("The agent arrived at tearminal state.")
                # print("Exit")
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
    df.to_csv(LOG_DIR + "{0}_{1}_{2}_fin.csv".format(agent.name, env.name, s))
    env.to_pickle(LOG_DIR + "mdp_{0}_{1}_{2}_fin.pkl".format(agent.name, env.name, s))
    agent.to_pickle(LOG_DIR + "agent_{0}_{1}_{2}_fin.pkl".format(agent.name, env.name, s))


def runs_episodes(_mdp, _agent, step=50, episode=100, seed=10):
    print("Running experiment: {0} in {1}".format(_agent.name, _mdp.name))
    for s in range(0, seed):
        np.random.seed(s)
        _agent.reset()
        print("-------- new seed: {0:02} starts --------".format(s))
        run_episodes(_mdp, _agent, step, episode, s)


if __name__ == "__main__":
    ###########################
    # GET THE BLOCKWORLD
    ###########################
    env = GraphWorld(
        name="graphworld",
        graphmap_path="../mdp/graphworld/map2.json"
    )
    env.set_step_cost(1.0)
    env.set_goal_reward(50.0)

    ###########################
    # GET THE AGENT
    ###########################
    qlearning = QLearningAgent(name="QLearning", actions=env.get_actions())

    ###########################
    # RUN
    ###########################
    runs_episodes(env, qlearning, step=25, episode=500)
