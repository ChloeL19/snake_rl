# before testing run this: pip install -e c_snake_gym
# import gym

# env = gym.make('c_snake:c_snake-v0')
# game_controller = env.controller 
# snakes_array = game_controller.snakes
# snake = snakes_array[0]

# # We will start by implementing a DQN architecture
# # This seems kinda good as a starting place for pytorch: https://github.com/dusty-nv/jetson-reinforcement/blob/master/python/gym-DQN.py
# # This is the official OpenAI baseline: https://github.com/openai/baselines/tree/master/baselines/deepq

# # File structure:
# # DQN.py --> contains the architecture for the DQN
# # trainer.py --> contains functions necessary for training
# # DQN_in_env.py --> contains training loop and testing loop
# # utils.py --> idk, utils functions and stuff
# # maybe later add replay_buffer.py

# for i in range(30):
# 	action = snake.UP
# 	obs, reward, _, _ = env.step(action)
# 	env.render()


#print(env.viewer)

## video test

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FFMpegWriter

def update_line(num, data, line):
    line.set_data(data[..., :num])
    return line,

# Fixing random state for reproducibility
np.random.seed(19680801)


# Set up formatting for the movie files
Writer = animation.writers['ffmpeg']
writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)


fig1 = plt.figure()

data = np.random.rand(2, 25)
l, = plt.plot([], [], 'r-')
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.xlabel('x')
plt.title('test')
line_ani = animation.FuncAnimation(fig1, update_line, 25, fargs=(data, l),
                                   interval=50, blit=True)
line_ani.save('lines.mp4', writer=writer)

fig2 = plt.figure()

x = np.arange(-9, 10)
y = np.arange(-9, 10).reshape(-1, 1)
base = np.hypot(x, y)
ims = []
for add in np.arange(15):
    ims.append((plt.pcolor(x, y, base + add, norm=plt.Normalize(0, 30)),))

im_ani = animation.ArtistAnimation(fig2, ims, interval=50, repeat_delay=3000,
                                   blit=True)
im_ani.save('im.mp4', writer=writer)

fig3 = plt.figure()
metadata = dict(title='Movie Test', artist='Matplotlib',
                comment='Movie support!')
writer = FFMpegWriter(fps=15, metadata=metadata)
l, = plt.plot([], [], 'k-o')

plt.xlim(-5, 5)
plt.ylim(-5, 5)

x0, y0 = 0, 0

with writer.saving(fig3, "writer_test.mp4", 100):
    for i in range(100):
        x0 += 0.1 * np.random.randn()
        y0 += 0.1 * np.random.randn()
        l.set_data(x0, y0)
        writer.grab_frame()