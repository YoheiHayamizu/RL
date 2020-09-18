import exe.exeutils
from mdp.graphworld.graphworld import GraphWorld
from agent.qlearning import QLearningAgent


if __name__ == "__main__":
    opts = exe.exeutils.parse_options()
    ###########################
    # GET THE BLOCKWORLD
    ###########################
    env = GraphWorld(
        name="graphworld",
        graphmap_path="../mdp/graphworld/map.json"
    )
    env.set_step_cost(0.0)
    env.set_goal_reward(1.0)
    env.set_stack_cost(0.0)

    ###########################
    # GET THE AGENT
    ###########################
    qlearning = QLearningAgent(name="Q-Learning", actions=env.get_executable_actions())

    ###########################
    # RUN
    ###########################
    exe.exeutils.runs_episodes(env, qlearning, step=opts.iters, episode=opts.episodes, seed=opts.seeds)
