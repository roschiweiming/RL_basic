"""
    Implementation of DDPG by pytorch
"""
import torch
import torch.nn as nn
import numpy as np


# 定义网络
class Net(nn.Module):
    def __init__(self, s_dim, a_dim):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(s_dim, 50)
        self.fc1.weight.data.normal_(0, 0.1)

        self.out = nn.Linear(50, a_dim)
        self.out.weight.data.normal_(0, 0.1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        action_value = self.out(x)
        return action_value


# 定义Double DQN
class DQN(object):
    def __init__(self,
                 s_dim,
                 a_dim,
                 memory_buffer,
                 learning_rate=0.001,
                 gamma=0.9,  # reward discount
                 e_greedy=0.9,
                 batch_size=32,
                 target_replace_iter=100
                 ):

        self.s_dim = s_dim
        self.a_dim = a_dim
        self.memory_buffer = memory_buffer
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.e_greedy = e_greedy
        self.batch_size = batch_size
        self.target_replace_iter = target_replace_iter
        self.learn_step_counter = 0
        self.eval_net, self.target_net = Net(self.s_dim, self.a_dim), Net(self.s_dim, self.a_dim)
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=self.learning_rate)
        self.loss_func = nn.MSELoss()

    def choose_action(self, state):
        state = torch.unsqueeze(torch.FloatTensor(state), 0)  # 维度改变 pytorch通过batch计算的
        if np.random.uniform() > self.e_greedy:
            action_value = self.eval_net.forward(state).detach()
            action = torch.max(action_value, 1)[1].numpy()
            action = action[0]
        else:
            action = np.random.randint(0, self.a_dim)
        return action

    def action_best(self, state):
        state = torch.unsqueeze(torch.FloatTensor(state), 0)  # 维度改变 pytorch通过batch计算的
        action_value = self.eval_net.forward(state).detach()
        action = torch.max(action_value, 1)[1].numpy()
        action = action[0]
        return action

    def learn(self):
        # update target network
        if self.learn_step_counter % self.target_replace_iter == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        s_sample, a_sample, r_sample, new_s_sample, done_sample = self.memory_buffer.sample(self.batch_size)

        s_sample = torch.from_numpy(s_sample)
        a_sample = torch.from_numpy(a_sample.astype(int))
        r_sample = torch.from_numpy(r_sample)
        new_s_sample = torch.from_numpy(new_s_sample)
        done_sample = torch.from_numpy(done_sample)

        # print("a_sample", a_sample.shape)

        # 通过评估网络输出32行每个s_sample对应的一系列动作值，然后.gather(1, a_sample)代表对每行对应索引b_a的Q值提取进行聚合
        q_eval = self.eval_net(s_sample).gather(1, a_sample)
        q_next = self.target_net(new_s_sample).detach()
        q_target = r_sample + (1 - done_sample) * self.gamma * q_next.max(1)[0].view(self.batch_size, 1)
        # q_next.max(1)[0]表示只返回每一行的最大值，不返回索引(长度为32的一维张量)；.view()表示把前面所得到的一维张量变成(BATCH_SIZE, 1)的形状；最终通过公式得到目标值
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.e_greedy >= 0.1:
            self.e_greedy *= 0.995
        else:
            self.e_greedy = 0.1

    def save_models(self, episode_count):
        torch.save(self.eval_net.state_dict(), './models/' + str(episode_count) + '_dqn.pt')
        print('********** Models saved **********')

    def load_models(self, episode_count):
        self.eval_net.load_state_dict(torch.load('./models/' + str(episode_count) + '_dqn.pt'))
        self.target_net.load_state_dict(self.eval_net.state_dict())
        print('********** Models load **********')