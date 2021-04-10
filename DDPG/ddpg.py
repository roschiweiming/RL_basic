"""
    Implementation of DDPG by pytorch
"""

import torch
import torch.nn as nn
import numpy as np
from collections import deque
import random


# -- Memory Buffer -- #
class MemoryBuffer:
    def __init__(self, size):
        self.buffer = deque(maxlen=size)
        self.maxsize = size
        self.len = 0    # current size

    def sample(self, count):
        count = min(count, self.len)
        batch = random.sample(self.buffer, count)

        s_array = np.float32([array[0] for array in batch])
        a_array = np.float32([array[1] for array in batch])
        r_array = np.float32([array[2] for array in batch])
        new_s_array = np.float32([array[3] for array in batch])
        done_array = np.float32([array[4] for array in batch])

        return s_array, a_array, r_array, new_s_array, done_array

    def len(self):
        return self.len

    def store_transition(self, s, a, r, new_s, done):
        transition = (s, a, [r], new_s, [done])
        self.len += 1
        if self.len > self.maxsize:
            self.len = self.maxsize
        self.buffer.append(transition)


# -- Update target network -- #
def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data*(1.0 - tau) + param.data*tau)


def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


# -- Critic -- #
class Actor(nn.Module):
    def __init__(self, s_dim, a_dim, a_bound):
        super(Actor, self).__init__()
        self.a_bound = a_bound      # Represents the limitation of action

        self.fa1 = nn.Linear(s_dim, 60)
        self.fa1.weight.data.normal_(0, 0.1)  # initialization

        self.fa2 = nn.Linear(60, a_dim)
        self.fa2.weight.data.normal_(0, 0.1)  # initialization

    def forward(self, state):
        x = torch.relu(self.fa1(state))
        x = torch.tanh(self.fa2(x))     # map to [-1, 1]
        action = x * self.a_bound       # map to [- action_bound, + action_bound]
        return action


# -- Critic -- #
class Critic(nn.Module):
    def __init__(self, s_dim, a_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(s_dim, 30)
        self.fc1.weight.data.normal_(0, 0.1)   # initialization

        self.fa1 = nn.Linear(a_dim, 30)
        self.fa1.weight.data.normal_(0, 0.1)   # initialization

        self.fca1 = nn.Linear(60, 60)
        self.fca1.weight.data.normal_(0, 0.1)  # initialization

        self.fca2 = nn.Linear(60, 1)
        self.fca2.weight.data.normal_(0, 0.1)  # initialization

    def forward(self, state, action):
        xs = torch.relu(self.fc1(state))
        xa = torch.relu(self.fa1(action))
        x = torch.cat((xs, xa), dim=1)
        x = torch.relu(self.fca1(x))
        action_value = self.fca2(x)
        return action_value


# -- DDPG Agent -- #
class DDPG(object):
    def __init__(
            self,
            state_dimension,
            action_dimension,
            action_bound,
            memory_buffer,
            learning_rate_actor=0.0001,  # learning rate for actor
            learning_rate_critic=0.0002,  # learning rate for critic
            batch_size=32,
            gamma=0.9,  # reward discount
            tau=0.01   # soft replacement
    ):
        self.state_dim = state_dimension
        self.action_dim = action_dimension
        self.action_bound = action_bound
        self.memory_buffer = memory_buffer
        self.LR_A = learning_rate_actor
        self.LR_C = learning_rate_critic
        self.batch_size = batch_size
        self.gamma = gamma
        self.TAU = tau

        self.actor = Actor(self.state_dim, self.action_dim, self.action_bound)
        self.target_actor = Actor(self.state_dim, self.action_dim, self.action_bound)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), self.LR_A)

        self.critic = Critic(self.state_dim, self.action_dim)
        self.target_critic = Critic(self.state_dim, self.action_dim)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), self.LR_C)
        self.loss_td = nn.MSELoss()

        hard_update(self.target_actor, self.actor)
        hard_update(self.target_critic, self.critic)

    def choose_action(self, state):
        state = torch.unsqueeze(torch.FloatTensor(state), 0)    # 维度改变 pytorch通过batch计算的
        return self.actor.forward(state)[0].detach()
       
    def learn(self):
        s_sample, a_sample, r_sample, new_s_sample, done_sample = self.memory_buffer.sample(self.batch_size)
        
        s_sample = torch.from_numpy(s_sample)
        a_sample = torch.from_numpy(a_sample)
        r_sample = torch.from_numpy(r_sample)
        new_s_sample = torch.from_numpy(new_s_sample)
        done_sample = torch.from_numpy(done_sample)

        # -- optimize critic  -- #
        a_target = self.target_actor.forward(new_s_sample).detach()
        next_value = self.target_critic.forward(new_s_sample, a_target).detach()
        y_expected = r_sample + (1 - done_sample)*self.gamma*next_value
        y_predicted = self.critic.forward(s_sample, a_sample)
        td_error = self.loss_td(y_expected, y_predicted)

        self.critic_optimizer.zero_grad()
        td_error.backward()
        self.critic_optimizer.step()

        # -- optimize actor   -- #
        pred_a_sample = self.actor.forward(s_sample)
        pred_value = self.critic.forward(s_sample, pred_a_sample)
        loss_actor = -torch.mean(pred_value)

        self.actor_optimizer.zero_grad()
        loss_actor.backward()
        self.actor_optimizer.step()

        soft_update(self.target_actor, self.actor, self.TAU)
        soft_update(self.target_critic, self.critic, self.TAU)

    def save_models(self, episode_count):
        torch.save(self.actor.state_dict(), './models/' + str(episode_count) + '_actor.pt')
        torch.save(self.critic.state_dict(), './models/' + str(episode_count) + '_critic.pt')
        print('********** Models saved **********')

    def load_models(self, episode):
        self.actor.load_state_dict(torch.load('./models/' + str(episode) + '_actor.pt'))
        self.critic.load_state_dict(torch.load('./models/' + str(episode) + '_critic.pt'))
        hard_update(self.target_actor, self.actor)
        hard_update(self.target_critic, self.critic)
        print('********** Models load **********')
