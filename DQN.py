# Deep Q-learning with experience replay
# implementing paper from here: https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf

import numpy as np
import tensorflow as tf 
from tensorflow import keras
import gym
import queue
from network import ActionValueFunction
import random
# for recording final performance
import pyautogui 
import cv2 
import numpy as np
# import matplotlib
# import matplotlib.pyplot as plt
# import matplotlib.animation as animation
# from celluloid import Camera
# camera = Camera(fig) #  https://pypi.org/project/celluloid/
# fig = plt.figure()

env = gym.make('c_snake:c_snake-v0')

# global variables 
NUM_EPISODES = 1 #1000
TIMESTEPS = 100 #100 # the max number of timesteps per episode (otherwise episode ends
				# when a terminal state is reached)
EPSILON = 0.3
MINIBATCH_SIZE = 28
DISPLAY_TIMESTEPS = 28 # displaying the agent's new behavior

# initialize experience replay
experience_replay = queue.LifoQueue(maxsize=NUM_EPISODES*TIMESTEPS)

# initialize action-value function
q_func = ActionValueFunction(env.last_obs.shape, env.action_space.n)

# fill the experience replay memory buffer with at least MINIBATCH_SIZE
# number of transitions
# FIXME: timesteps should really be the maximum number of timesteps in each
# episode 
def run_agent(num_timesteps, record=False):
	# frames = []
	_ = env.reset()
	if record:
		# Specify resolution 
		resolution = (1920, 1200) 
		#resolution = (int(cap.get(3)), int(cap.get(4)))
		# Specify video codec 
		codec = cv2.VideoWriter_fourcc(*'XVID') 
		# Specify name of Output file 
		filename = "Recording.avi"
		# Specify frames rate. We can choose  
		# any value and experiment with it 
		fps = 60.0
		# Creating a VideoWriter object 
		out = cv2.VideoWriter(filename, codec, fps, resolution)

	for i in range(num_timesteps):
		current_state = env.last_obs/255
		action = int(q_func.get_best_action(current_state))
		obs, reward, done, _ = env.step(action)
		next_state = obs/255
		experience_replay.put((current_state, action, reward, next_state, done))
		env.render(record=record)
		if record:
			# Take screenshot using PyAutoGUI 
			img = pyautogui.screenshot() 
			# Convert the screenshot to a numpy array 
			frame = np.array(img) 
			# Convert it from BGR(Blue, Green, Red) to 
			# RGB(Red, Green, Blue) 
			frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 
			# Write it to the output file 
			out.write(frame)
		# 	frames.append([plt.imshow(env.last_obs, animated=True)])
		if done:
			# if record:
			# 	env.saveVideo(name="timestep_{}".format(i))
			_ = env.reset()
	if record:
		# Release the Video writer 
		out.release() 
	# 	env.saveVideo(name="finalmovie")
	
	# if record:
	# 	# Set up formatting for the movie files
	# 	Writer = animation.writers['ffmpeg']
	# 	writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)
	# 	ani = animation.ArtistAnimation(fig, frames, interval=50, blit=True,
	# 								repeat_delay=1000)
	# 	ani.save('movie.mp4', writer=writer)
	# 	print(frames)

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
			experience_replay.put((current_state, action, reward, next_state, done))

			# sample random minibatch from experience replay and train
			mini_batch = random.sample(list(experience_replay.queue), MINIBATCH_SIZE)
			for (curr_s, a, r, next_s, d) in mini_batch: # is it really a good idea to minibatch??
				loss = q_func.gradient_step(curr_s, next_s, a, r, d)
				print(loss)
			env.render()
		else:
			break


# how to save the following as a video that can be replayed?
# start screen recording

run_agent(DISPLAY_TIMESTEPS, record=True)

# okay not convinced that went well but I hope this works
# testing
# for q in iter(experience_replay.get, None):
# 	print(q)
# make sure to divide pixel data by 255!!

