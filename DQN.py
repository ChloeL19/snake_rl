# Deep Q-learning with experience replay
# implementing paper from here: https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf

import numpy as np
import tensorflow as tf 
from tensorflow import keras
import gym
import queue
from network import ActionValueFunction
import random
import numpy as np
from matplotlib.animation import FFMpegWriter
import matplotlib.pyplot as plt

env = gym.make('c_snake:c_snake-v0')

# global variables slwetwe
NUM_EPISODES = 1000 #1000
TIMESTEPS = 100 #100 # the max number of timesteps per episode (otherwise episode ends
				# when a terminal state is reached)
EPSILON = 0.3
MINIBATCH_SIZE = 25
DISPLAY_TIMESTEPS = 100 # displaying the agent's new behavior
REC_FREQ = 10 # recording frequency; number of epochs after which to record
MEMORY_LENGTH = 500 # number of timesteps to keep in replay memory

# initialize experience replay
experience_replay = queue.LifoQueue(maxsize=NUM_EPISODES*TIMESTEPS)
# Did not work: experience_replay = queue.LifoQueue(maxsize=MEMORY_LENGTH)
#experience_replay = queue.Queue(maxsize=MEMORY_LENGTH)

# initialize action-value function
q_func = ActionValueFunction(env.last_obs.shape, env.action_space.n)

# fill the experience replay memory buffer with at least MINIBATCH_SIZE
# number of transitions
# FIXME: timesteps should really be the maximum number of timesteps in each
# episode 
def run_agent(num_timesteps, record=False, title="recording_test.mp4"):
	done = False
	_ = env.reset()
	if record:
		print("recording a test run . . . ")
		fig = plt.figure()
		metadata = dict(title='Movie Test', artist='Matplotlib',
						comment='Movie support!')
		writer = FFMpegWriter(fps=15, metadata=metadata)
		with writer.saving(fig, title, 100):
			for i in range(num_timesteps):
				current_state = env.last_obs/255
				action = int(q_func.get_best_action(current_state))
				obs, reward, done, _ = env.step(action)
				#next_state = obs/255
				#experience_replay.put((current_state, action, reward, next_state, done))
				plt.imshow(obs)
				writer.grab_frame()
				env.render(record=record)
				#i += 1
				if done:
					_ = env.reset()
				
	else:
		for i in range(num_timesteps):
			current_state = env.last_obs/255
			action = int(q_func.get_best_action(current_state))
			obs, reward, done, _ = env.step(action)
			next_state = obs/255
			try:
				experience_replay.put((current_state, action, reward, next_state, done))
			except queue.Full as e:
				import pdb; pdb.set_trace();
			env.render(record=record)
			#i += 1
			if done:
				_ = env.reset()

run_agent(MINIBATCH_SIZE)
#import pdb; pdb.set_trace();

# perform training loop described in Algorithm 1
# https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf
for i in range(NUM_EPISODES):
	_ = env.reset()
	done = False
	for t in range(TIMESTEPS):
		if not done: 
			current_state = env.last_obs/255
			if np.random.rand() < EPSILON: # select random action with probability epsilon
				action = random.randrange(0,4)
			else:
				action = int(q_func.get_best_action(env.last_obs))
			
			obs, reward, done, _ = env.step(action)
			next_state = obs/255
			try:
				experience_replay.put((current_state, action, reward, next_state, done))
			except queue.Full as e:
				import pdb; pdb.set_trace();
			# if len(list(experience_replay.queue)) >= MEMORY_LENGTH:
			# 	_ = experience_replay.get() # confirm this

			# sample random minibatch from experience replay and train
			mini_batch = random.sample(list(experience_replay.queue), MINIBATCH_SIZE)
			for (curr_s, a, r, next_s, d) in mini_batch: # is it really a good idea to minibatch??
				loss = q_func.gradient_step(curr_s, next_s, a, r, d)
				print(loss)
			env.render()
		else:
			break
	if i % REC_FREQ == 0:
		run_agent(DISPLAY_TIMESTEPS, title="episode_{}.mp4".format(i), record=True)

run_agent(DISPLAY_TIMESTEPS, record=True)

