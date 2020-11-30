# Deep Q-learning with experience replay
# implementing paper from here: https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf

import numpy as np
import tensorflow as tf 
from tensorflow import keras
import gym

env = gym.make('c_snake:c_snake-v0')

# global variables 
NUM_EPISODES = 1000
TIMESTEPS = 100 # the max number of timesteps per episode (otherwise episode ends
				# when a terminal state is reached)

# how do I ensure I start with a random initial state each time? also 
# is that sufficient?

# initialize experience replay
experience_replay = np.zeros((NUM_EPISODES*TIMESTEPS))

# initialize action-value function
print(env.grid_size)