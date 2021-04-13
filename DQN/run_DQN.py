"""
    run dqn
"""
import gym
from DQN.DQN import *
from Memory.MemoryBuffer import *

# -- hyper parameters -- #
MAX_EPISODES = 1000
MAX_EP_STEPS = 200
LR = 0.001  # learning rate
GAMMA = 0.9  # reward discount
EPSILON = 0.9
TARGET_REPLACE_ITER = 100
MEMORY_CAPACITY = 50000
BATCH_SIZE = 32
RENDER = False
ENV_NAME = 'CartPole-v0'
env = gym.make(ENV_NAME).unwrapped
N_ACTIONS = env.action_space.n
N_STATES = env.observation_space.shape[0]

if __name__ == '__main__':
    memory_buffer = MemoryBuffer(MEMORY_CAPACITY)
    dqn = DQN(N_STATES, N_ACTIONS, memory_buffer, LR, GAMMA, EPSILON, BATCH_SIZE, TARGET_REPLACE_ITER)

    for i in range(MAX_EPISODES + 1):
        s = env.reset()
        ep_reward = 0
        while True:
            if RENDER:
                env.render()
            a = dqn.choose_action(s)
            s_, r, done, info = env.step(a)

            memory_buffer.store_transition(s, a, r, s_, done)
            ep_reward += r

            s = s_
            if memory_buffer.len() > 2000:
                dqn.learn()
            if done:
                print('episode%s---reward_sum: %s' % (i, round(ep_reward, 2)))
                break

