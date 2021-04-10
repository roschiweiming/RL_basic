import numpy
import torch
import gym

ENV_NAME = 'Pendulum-v0'
env = gym.make(ENV_NAME)

s_dim = env.observation_space.shape[0]
a_dim = env.action_space.shape[0]
a_bound = env.action_space.high[0]

state = env.observation_space
action = env.action_space

env.reset()
observation, reward, done, _ = env.step(env.action_space.sample())

# print(s_dim)
# print(a_dim)
# print(a_bound)
# print(state)
# print(action)

print(observation)
print(reward)

torch.FloatTensor(observation)
print(observation)