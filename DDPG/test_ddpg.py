"""
    To test ddpg
"""
import gym
import matplotlib.pyplot as plt
from RL_basic.DDPG.ddpg import *

# -- hyper parameters -- #
MAX_EPISODES = 100
MAX_EP_STEPS = 200
MEMORY_CAPACITY = 50000
RENDER = True
LOAD_MODELS = True
load_episodes = 10000
ENV_NAME = 'Pendulum-v0'

if __name__ == '__main__':
    env = gym.make(ENV_NAME)
    env = env.unwrapped
    env.seed(1)
    s_dim = env.observation_space.shape[0]
    a_dim = env.action_space.shape[0]
    a_bound = env.action_space.high[0]
    print('state dimension: ', s_dim)
    print('action dimension: ', a_dim)
    print('limitation of action: ', a_bound)

    memory_buffer = MemoryBuffer(MEMORY_CAPACITY)
    ddpg = DDPG(s_dim, a_dim, a_bound, memory_buffer)

    if LOAD_MODELS:
        ddpg.load_models(load_episodes)

    realtime_reward = []
    average_reward = []

    for i in range(MAX_EPISODES + 1):
        s = env.reset()
        ep_reward = 0
        for j in range(MAX_EP_STEPS):
            env.render()

            a = ddpg.choose_action(s)
            s_, r, done, info = env.step(a)
            s = s_
            ep_reward += r
            if j == MAX_EP_STEPS - 1:
                print('Episode:', i, ' Reward: %i' % int(ep_reward))
                realtime_reward.append(ep_reward)
                average_reward.append(np.mean(realtime_reward))
                break
    plt.plot(np.arange(len(average_reward)), average_reward, 'b-', label='average_reward', alpha=1)
    plt.plot(np.arange(len(realtime_reward)), realtime_reward, 'g-', label='RealTime_reward', alpha=0.3)
    plt.ylabel('Reward')
    plt.xlabel('training steps')
    plt.title('Test Reward curve')
    plt.legend()
    plt.savefig('./result/Test Reward_1.png')  # 保存图片
    plt.show()
