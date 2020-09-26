#!/bin/bash
export PYTHONPATH=~/Documents/researches/RL/

## Blockworld
#python ./run_blockworld.py
#python ../utils/graphics.py --mdp blockworld
#
## Gridworld
#python run_gridworld.py
#python ../utils/graphics.py --mdp gridworld

# Graphworld
python run_graphworld.py -s 1 -e 5000 -i 25
python ../utils/graphics.py --mdp gridmap -w 50
