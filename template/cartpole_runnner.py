# import the gym stuff
import gym
# import other stuff
import random
import numpy as np
# import own classes
from simulated_annealing import SA

env = gym.make('CartPole-v0')

epochs = 10000
steps = 200
scoreTarget = 200
starting_temp = 1
final_temp = 0.001

sa = SA(len(env.observation_space.high), env.action_space.n, 10, env, steps, epochs, scoreTarget = scoreTarget, starting_temp = starting_temp, final_temp = final_temp, max_change= 1.0)

# network size for the agents
sa.initAgent([4])

sa.sa()