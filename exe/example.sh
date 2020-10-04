#!/bin/bash
export PYTHONPATH=~/Documents/researches/RL/

# Blockworld
python ../mdp/blockworld/blockworld.py --mdpName blockworld -s 1 -e 1000 -i 50
python ../utils/graphics.py --mdp blockworld -w 100

# Gridworld
python ../mdp/gridworld/gridworld.py --mdpName gridworld -s 1 -e 1000 -i 50
python ../utils/graphics.py --mdp gridworld -w 100

# Graphworld
python ../mdp/graphworld/graphworld.py --mdpName graphworld2 -s 1 -e 1000 -i 50
python ../utils/graphics.py --mdp graphworld2 -w 100
