"""
    run dqn
"""
import gym
import argparse
from DQN.dqn import *
from Memory.MemoryBuffer import *

parser = argparse.ArgumentParser()
parser.add_argument('--mode', default='test', type=str)
args = parser.parse_args()

# -- hyper parameters -- #
MAX_EPISODES = 300
MAX_EP_STEPS = 200
LR = 0.001  # learning rate
GAMMA = 0.9  # reward discount
EPSILON = 0.9
TARGET_REPLACE_ITER = 100
MEMORY_CAPACITY = 2000
BATCH_SIZE = 32
load_episodes = 300
RENDER = False
ENV_NAME = 'CartPole-v0'
env = gym.make(ENV_NAME).unwrapped
N_ACTIONS = env.action_space.n
N_STATES = env.observation_space.shape[0]

if __name__ == '__main__':
    memory_buffer = MemoryBuffer(MEMORY_CAPACITY)
    dqn = DQN(N_STATES, N_ACTIONS, memory_buffer, LR, GAMMA, EPSILON, BATCH_SIZE, TARGET_REPLACE_ITER)

    if args.mode == 'train':
        for i in range(MAX_EPISODES + 1):
            s = env.reset()
            ep_reward = 0
            while True:
                if RENDER:
                    env.render()
                a = dqn.choose_action(s)
                s_, r, done, info = env.step(a)

                # 修改奖励 (不修改也可以，修改奖励只是为了更快地得到训练好的摆杆)
                x, x_dot, theta, theta_dot = s_
                r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
                r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
                new_r = r1 + r2

                memory_buffer.store_transition(s, a, new_r, s_, done)
                ep_reward += r

                s = s_
                if memory_buffer.len > 100:
                    dqn.learn()
                if done:
                    print('episode %s---reward_sum: %s---epsilon: %s' % (i, round(ep_reward, 2), dqn.e_greedy))
                    break
            if i % 100 == 0 and i > 0:
                dqn.save_models(i)

    elif args.mode == 'test':
        dqn.load_models(load_episodes)
        for i in range(MAX_EPISODES + 1):
            s = env.reset()
            ep_reward = 0
            while True:
                env.render()
                a = dqn.action_best(s)
                s_, r, done, info = env.step(a)
                ep_reward += r
                if done:
                    print('episode %s---reward_sum: %s' % (i, round(ep_reward, 2)))
                    break
                s = s_
