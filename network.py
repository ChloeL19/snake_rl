import numpy as np
import tensorflow as tf 
from tensorflow import keras

class ActionValueFunction:

	"""
	This is the action-value function for the DQN. 
	I guess the preprocessing function Phi is just the identity function.
	"""

	def __init__(self, grid_size, action_space, fc_layers=1, conv_layers=0):
		self.input_shape = grid_size
		self.output_shape = action_space
		# not actually used yet
		self.fc_layers = fc_layers
		self.conv_layers = conv_layers

		# define the model here
		# Question: should the output of the value network
		# be constrained by the properties of the softmax function???
		self.model = tf.keras.models.Sequential([
			tf.keras.Dense(units=self.output_shape, activation='sigmoid', 
				input_shape=self.input_shape),
			tf.keras.Dense(units=self.output_shape, activation='softmax')
		])

	def get_best_action(self, x):
		"""
		Returns the action with the highest value based on the output
		of the neural network.
		x : input to the neural network, i.e. the units in the grid
		"""
		return np.argmax(self.model(x))

	def gradient_step(self, y, state, action):
		"""
		computes the gradient of the cost function with respect to the parameters
		of the Q value network.
		- y: the target
		- state: a state selected from the experience replay memory
		- action: action taken in the state 
		"""

