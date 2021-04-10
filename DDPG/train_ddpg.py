"""
    To train ddpg
"""
import gym
import matplotlib.pyplot as plt
from RL_basic.DDPG.ddpg import *

# -- hyper parameters -- #
MAX_EPISODES = 10000
MAX_EP_STEPS = 200
LR_A = 0.0001  # learning rate for actor
LR_C = 0.0002  # learning rate for critic
GAMMA = 0.9  # reward discount
TAU = 0.01  # soft replacement
MEMORY_CAPACITY = 50000
BATCH_SIZE = 32
RENDER = False
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
    ddpg = DDPG(s_dim,
                a_dim,
                a_bound,
                memory_buffer,
                learning_rate_actor=LR_A,
                learning_rate_critic=LR_C,
                batch_size=BATCH_SIZE,
                gamma=GAMMA,
                tau=TAU
                )

    var = 3  # control exploration

    total_reward = []
    average_reward = []

    for i in range(MAX_EPISODES + 1):
        s = env.reset()
        ep_reward = 0
        for j in range(MAX_EP_STEPS):
            if RENDER:
                env.render()

            # add randomness to action selection for exploration, and change tensor to array
            a = ddpg.choose_action(s)
            a = np.clip(np.random.normal(a, var), -a_bound, a_bound)
            s_, r, done, _ = env.step(a)

            memory_buffer.store_transition(s, a, r / 10, s_, done)

            if memory_buffer.len >= MEMORY_CAPACITY:
                var *= .9995  # decay the action randomness
                if var <= 0.001 and i < 0.95 * MAX_EPISODES:
                    var = 0.001
                ddpg.learn()

            s = s_
            ep_reward += r
            if j == MAX_EP_STEPS - 1:
                print('Episode: ', i, 'Reward: %i' % int(ep_reward), 'Explore: %.5f' % var, 'memory_buffer: ',
                      memory_buffer.len())
                total_reward.append(ep_reward)  # record reward of each episodes
                average_reward.append(np.mean(total_reward))  # record mean reward of each episodes
                break
        if i % 100 == 0:
            ddpg.save_models(i)

    plt.plot(np.arange(len(average_reward)), average_reward, 'b-', label='average_reward', alpha=1)
    plt.plot(np.arange(len(total_reward)), total_reward, 'g-', label='RealTime_reward', alpha=0.3)
    plt.ylabel('Reward')
    plt.xlabel('training steps')
    plt.title('Reward curve(LR_A = 0.0001,LR_C = 0.0002)')
    plt.legend()
    plt.savefig('./result/Total Reward_5.png')  # 保存图片
    plt.show()
