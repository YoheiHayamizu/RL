import exe.exeutils
from mdp.gridworld.gridworld import GridWorld
from agent.qlearning import QLearningAgent

if __name__ == "__main__":
    opts = exe.exeutils.parse_options()
    ###########################
    # GET THE BLOCKWORLD
    ###########################
    from mdp.gridworld.map2 import MAP2
    env = GridWorld(name=opts.mdpName,
                    gridmap=MAP2)
    env.set_step_cost(0.0)
    env.set_goal_reward(1.0)

    ###########################
    # GET THE AGENT
    ###########################
    qlearning = QLearningAgent(name="Q-Learning", actions=env.get_executable_actions())

    ###########################
    # RUN
    ###########################
    exe.exeutils.runs_episodes(env, qlearning, step=opts.iters, episode=opts.episodes, seed=opts.seeds)
