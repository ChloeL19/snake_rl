import numpy as np
import tensorflow as tf 
from tensorflow import keras

class ActionValueFunction:

	"""
	This is the action-value function for the DQN. 
	I guess the preprocessing function Phi is just the identity function.
	Just kidding. Preprocessing involves normalizing pixel values, so
	dividing by 255.
	"""

	def __init__(self, grid_size, action_space, fc_layers=2, conv_layers=0):
		self.input_shape = grid_size
		self.output_shape = action_space
		# not actually used yet
		self.fc_layers = fc_layers
		self.conv_layers = conv_layers
		kernel_size = (35, 35)

		self.optimizer = tf.keras.optimizers.Adam(learning_rate=10e-5)

		# define the model here
		# Question: should the output of the value network
		# be constrained by the properties of the softmax function???
		self.model = tf.keras.models.Sequential([
			tf.keras.layers.Conv2D(filters=1, kernel_size=kernel_size, activation='tanh', 
				input_shape=self.input_shape),
			tf.keras.layers.Flatten(),
			tf.keras.layers.Dense(units=self.output_shape, activation='sigmoid')
		])

	def predict(self, x):
		"""
		Returns the raw model predictions consisting of a value for each
		action in the action space. 
		x : input to the neural network, i.e. the image of the game space
		"""
		x = np.expand_dims(x, axis=0)
		return self.model.predict(x)

	def get_best_action(self, x):
		"""
		Returns the action with the highest value based on the output
		of the neural network.
		x : input to the neural network, i.e. the units in the grid
		"""
		# add dimension to x to work with keras
		x = np.expand_dims(x, axis=0)
		return np.argmax(self.model.predict(x))

	#@tf.function
	def gradient_step(self, state, next_state, action, reward, done, lam=0.7):
		"""
		computes the gradient of the cost function with respect to the parameters
		of the Q value network.
		- state: a state selected from the experience replay memory, which is 
				a matrix of size grid_size
		- next_state: the state that directly follows the previous state
		- action: action taken in the state 
		- reward: the reward for the current state
		- done: a boolean value indicating whether or not this is a terminal state
		- lamb: reward discount factor
		"""

		# calculate the target y, which is either the reward in the state or the estimate thereof
		# I'm going to vectorize the target, which I think is the only
		# way to make this thing differentiable
		if done:
			y = reward*tf.ones((self.output_shape))
		else:
			next_state = tf.expand_dims(next_state, axis=0)
			next_reward = self.model(next_state)
			y = reward + lam*next_reward
		with tf.GradientTape() as tape:
			# tape.watch(tf.convert_to_tensor(mask))
			state = tf.expand_dims(state, axis=0)
			output_vals = self.model(state)
			loss = (y - output_vals)**2 # is indexing a legal operation here???

		grads = tape.gradient(loss, self.model.trainable_variables)
		self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

		return np.mean(loss) # sketchy