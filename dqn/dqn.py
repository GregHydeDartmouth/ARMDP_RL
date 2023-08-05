import torch
import random
import numpy as np
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

class QNet(nn.Module):
    def __init__(self, state_space, action_space, hidden_space=128):
        super(QNet, self).__init__()

        self.hidden_space = hidden_space
        self.state_space = state_space
        self.action_space = action_space

        self.Linear1 = nn.Linear(state_space, self.hidden_space)
        self.Linear2 = nn.Linear(self.hidden_space, action_space)

    def forward(self, x):
        hidden = F.relu(self.Linear1(x))
        return self.Linear2(hidden)

class DQN(nn.Module):
    def __init__(self, state_space, action_space, hidden_space=128, lr=1e-3, tau=0.001, batch=32, discount=0.95, capacity=100000, seed=1):
        super(DQN, self).__init__()
        np.random.seed(seed)
        random.seed(seed)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.state_space = state_space
        self.action_space = action_space
        self.hidden_space = hidden_space
        self.lr = lr
        self.tau = tau
        self.batch = batch
        self.discount = discount
        self.capacity = capacity

        self.q_net = QNet(self.state_space, self.action_space, hidden_space=self.hidden_space).to(self.device)
        self.target_q_net = QNet(self.state_space, self.action_space, hidden_space=self.hidden_space).to(self.device)
        self.target_q_net.load_state_dict(self.q_net.state_dict())

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=self.lr)
        self.replay_buffer = ReplayBuffer(self.capacity)

    def act(self, state, epsilon=0.1):
        if random.random() < epsilon:
            return random.randint(0, self.action_space - 1)
        else:
            with torch.no_grad():
                state = torch.FloatTensor(state).to(self.device)
                q_values = self.q_net(state)
                return torch.argmax(q_values, dim=0).item()

    def step(self, state, action, reward, next_state, done):
        self.replay_buffer.push(state, action, reward, next_state, done)

    def train(self):
        if len(self.replay_buffer) > self.batch:
            states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch)
            states = torch.FloatTensor(states).to(self.device)
            actions = torch.tensor(actions, dtype = torch.int64).to(self.device)
            rewards = torch.FloatTensor(rewards).to(self.device)
            next_states = torch.FloatTensor(next_states).to(self.device)
            dones = torch.FloatTensor(dones).to(self.device)

            q_values = self.q_net(states)
            next_q_values = self.target_q_net(next_states)

            q_value = q_values.gather(1, actions)
            next_q_value, _ = torch.max(next_q_values, dim=1, keepdim=True)

            td_targets = rewards + self.discount * next_q_value * (1 - dones)
            loss = F.mse_loss(q_value, td_targets.detach())
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            for target_param, param in zip(self.target_q_net.parameters(), self.q_net.parameters()):
                target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)
