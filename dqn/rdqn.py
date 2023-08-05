import torch
import random
import numpy as np
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from collections import deque

class RQNet(nn.Module):
    def __init__(self, state_space, action_space, hidden_space):
        super(RQNet, self).__init__()

        self.state_space = state_space
        self.action_space = action_space
        self.hidden_space = hidden_space

        self.rnn = nn.RNN(self.state_space, self.hidden_space, batch_first=True)
        self.Linear1 = nn.Linear(self.hidden_space, self.action_space)

    def forward(self, x, h):
        x, h_prime = self.rnn(x, h)
        return self.Linear1(F.relu(x)), h_prime

class RDQN(object):
    def __init__(self, state_space, action_space, hidden_space=128, lr=1e-3, tau=0.001, discount=0.95, capacity=100000, seed=1):
        super(RDQN, self).__init__()
        np.random.seed(seed)
        random.seed(seed)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.state_space = state_space
        self.action_space = action_space
        self.hidden_space = hidden_space
        self.lr = lr
        self.tau = tau
        self.discount = discount
        self.capacity = capacity

        self.q_net = RQNet(self.state_space, self.action_space, hidden_space=self.hidden_space).to(self.device)
        self.target_q_net = RQNet(self.state_space, self.action_space, hidden_space=self.hidden_space).to(self.device)
        self.target_q_net.load_state_dict(self.q_net.state_dict())

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=self.lr)

        self.buffer = deque(maxlen=self.capacity)
        self.trajectory = []

    def get_init_hidden_state(self):
        return torch.zeros([1, self.hidden_space])

    def act(self, state, h, epsilon=0.1):
        if random.random() < epsilon:
            with torch.no_grad():
                state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                _, h_prime = self.q_net(state, h)
            return random.randint(0, self.action_space - 1), h_prime
        else:
            with torch.no_grad():
                state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values, h_prime = self.q_net(state, h)
                return torch.argmax(q_values, dim=1).item(), h_prime

    def step(self, state, action, reward, next_state, done):
        self.trajectory.append([state, action, reward, next_state, done])

    def reset(self):
        self.buffer.append(self.trajectory)
        self.trajectory = []

    def train(self):

        if len(self.buffer) >= 1:

            trajectory = random.choice(self.buffer)
            states, actions, rewards, next_states, dones = zip(*trajectory)
            states = torch.FloatTensor(np.array(states)).unsqueeze(0).to(self.device)
            actions = torch.LongTensor(np.array(actions)).unsqueeze(0).to(self.device)
            rewards = torch.FloatTensor(np.array(rewards)).unsqueeze(0).to(self.device)
            next_states = torch.FloatTensor(np.array(next_states)).unsqueeze(0).to(self.device)
            dones = torch.FloatTensor(np.array(dones)).unsqueeze(0).to(self.device)

            h = torch.zeros(1, 1, self.hidden_space).to(self.device)
            q_values, _ = self.q_net(states, h)

            h_target = torch.zeros(1, 1, self.hidden_space).to(self.device)
            next_q_values, _ = self.target_q_net(next_states, h_target)

            q_value = q_values.gather(2, actions)
            next_q_value, _ = torch.max(next_q_values, dim=2, keepdim=True)

            td_targets = rewards + self.discount * next_q_value * (1-dones)
            q_loss = F.mse_loss(q_value, td_targets.detach())

            self.optimizer.zero_grad()
            q_loss.backward()
            self.optimizer.step()

            for target_param, param in zip(self.target_q_net.parameters(), self.q_net.parameters()):
                target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
