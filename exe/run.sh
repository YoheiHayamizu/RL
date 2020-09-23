#!/bin/bash
export PYTHONPATH=~/Documents/researches/RL/
echo $PYTHONPATH

## Blockworld
#python ./run_blockworld.py
#python ../utils/graphics.py --mdp blockworld
#
## Gridworld
#python run_gridworld.py
#python ../utils/graphics.py --mdp gridworld

# Graphworld
python run_graphworld.py --mdp
python ../utils/graphics.py --mdp graphmap_uncertainty
#python ../utils/graphics.py --mdp stationary
