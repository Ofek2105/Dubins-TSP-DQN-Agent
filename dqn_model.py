import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque

class DQN(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, act_dim)
        )

    def forward(self, x):
        return self.net(x)

class DQNAgent:
    def __init__(self, obs_dim, act_dim):
        self.model = DQN(obs_dim, act_dim)
        self.target_model = DQN(obs_dim, act_dim)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()

        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        self.criterion = nn.MSELoss()

        self.memory = deque(maxlen=100_000)
        self.batch_size = 64
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.1
        self.update_target_freq = 1000
        self.step_count = 0

    def act(self, obs):
        obs = torch.tensor(obs, dtype=torch.float32)
        if random.random() < self.epsilon:
            return np.random.randint(0, self.model.net[-1].out_features, size=len(obs))
        with torch.no_grad():
            return self.model(obs).argmax(dim=1).cpu().numpy()

    def remember(self, s, a, r, s_, d):
        for i in range(len(s)):
            self.memory.append((s[i], a[i], r[i], s_[i], d[i]))

    def train_step(self):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        s, a, r, s_, d = zip(*batch)

        s = torch.tensor(s, dtype=torch.float32)
        a = torch.tensor(a, dtype=torch.int64).unsqueeze(1)
        r = torch.tensor(r, dtype=torch.float32).unsqueeze(1)
        s_ = torch.tensor(s_, dtype=torch.float32)
        d = torch.tensor(d, dtype=torch.float32).unsqueeze(1)

        q_vals = self.model(s).gather(1, a)
        with torch.no_grad():
            max_q_vals = self.target_model(s_).max(1, keepdim=True)[0]
            target = r + self.gamma * max_q_vals * (1 - d)

        loss = self.criterion(q_vals, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.step_count += 1
        if self.step_count % self.update_target_freq == 0:
            self.target_model.load_state_dict(self.model.state_dict())

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
