import exe.exeutils

from mdp.blockworld.blockworld import BlockWorld
from agent.qlearning import QLearningAgent


if __name__ == "__main__":
    opts = exe.exeutils.parse_options()
    ###########################
    # GET THE BLOCKWORLD
    ###########################
    env = BlockWorld(
        name="blockworld",
        width=5,
        height=5,
        init_loc=(0, 0),
        goal_loc=(4, 4),
        walls_loc=((3, 1), (3, 2), (3, 3), (0, 2), (1, 2), (1, 1),),
        holes_loc=()
    )

    ###########################
    # GET THE AGENT
    ###########################
    qlearning = QLearningAgent(name="Q-Learning", actions=env.get_executable_actions())

    ###########################
    # RUN
    ###########################
    exe.exeutils.runs_episodes(env, qlearning, step=opts.iters, episode=opts.episodes, seed=opts.seeds)
