# before testing run this: pip install -e c_snake_gym
import gym

env = gym.make('c_snake:c_snake-v0')
game_controller = env.controller 
snakes_array = game_controller.snakes
snake = snakes_array[0]

# We will start by implementing a DQN architecture
# This seems kinda good as a starting place for pytorch: https://github.com/dusty-nv/jetson-reinforcement/blob/master/python/gym-DQN.py
# This is the official OpenAI baseline: https://github.com/openai/baselines/tree/master/baselines/deepq

# File structure:
# DQN.py --> contains the architecture for the DQN
# trainer.py --> contains functions necessary for training
# DQN_in_env.py --> contains training loop and testing loop
# utils.py --> idk, utils functions and stuff
# maybe later add replay_buffer.py

for i in range(30):
	action = snake.UP
	obs, reward, _, _ = env.step(action)
	env.render()


#print(env.viewer)