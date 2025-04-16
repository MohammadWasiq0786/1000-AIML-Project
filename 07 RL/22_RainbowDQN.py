"""
Project 262. Rainbow DQN implementation
Description:
Rainbow DQN is a powerful deep reinforcement learning agent that combines six improvements over vanilla DQN into a single algorithm. These are:

Double DQN – reduces overestimation.

Dueling Networks – separates value and advantage estimation.

Prioritized Experience Replay – samples important experiences more frequently.

Multi-step learning – uses n-step returns for faster propagation.

Distributional RL – models full return distributions.

Noisy Nets – enables exploration via parameter noise instead of ε-greedy.

In this project, we’ll simulate a mini-Rainbow agent in a simple environment using PyTorch, focusing on combining key ideas (Double DQN + Dueling + Noisy Layers).

About:
✅ What It Does:
Uses Noisy Networks for learned exploration (instead of epsilon-greedy).

Applies Dueling Architecture to separate value from advantage.

Implements Double DQN by using separate target and online nets.

Can be extended with Prioritized Replay and Distributional Q.
"""
# Install dependencies:
# pip install gym torch numpy matplotlib
 
import gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import matplotlib.pyplot as plt
 
# Noisy linear layer for exploration
class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, sigma_init=0.5):
        super().__init__()
        self.mu_w = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.sigma_w = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.register_buffer("epsilon_w", torch.FloatTensor(out_features, in_features))
 
        self.mu_b = nn.Parameter(torch.FloatTensor(out_features))
        self.sigma_b = nn.Parameter(torch.FloatTensor(out_features))
        self.register_buffer("epsilon_b", torch.FloatTensor(out_features))
 
        self.reset_parameters()
        self.reset_noise()
 
    def reset_parameters(self):
        mu_range = 1 / np.sqrt(self.mu_w.size(1))
        self.mu_w.data.uniform_(-mu_range, mu_range)
        self.sigma_w.data.fill_(0.017)
        self.mu_b.data.uniform_(-mu_range, mu_range)
        self.sigma_b.data.fill_(0.017)
 
    def reset_noise(self):
        self.epsilon_w.normal_()
        self.epsilon_b.normal_()
 
    def forward(self, x):
        if self.training:
            weight = self.mu_w + self.sigma_w * self.epsilon_w
            bias = self.mu_b + self.sigma_b * self.epsilon_b
        else:
            weight = self.mu_w
            bias = self.mu_b
        return torch.nn.functional.linear(x, weight, bias)
 
# Dueling DQN with Noisy Layers
class RainbowDQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.feature = nn.Sequential(nn.Linear(state_dim, 128), nn.ReLU())
 
        self.advantage = nn.Sequential(
            NoisyLinear(128, 128), nn.ReLU(),
            NoisyLinear(128, action_dim)
        )
 
        self.value = nn.Sequential(
            NoisyLinear(128, 128), nn.ReLU(),
            NoisyLinear(128, 1)
        )
 
    def forward(self, x):
        features = self.feature(x)
        adv = self.advantage(features)
        val = self.value(features)
        return val + adv - adv.mean(dim=1, keepdim=True)
 
    def reset_noise(self):
        for layer in self.modules():
            if isinstance(layer, NoisyLinear):
                layer.reset_noise()
 
# Experience replay
class ReplayBuffer:
    def __init__(self, size=10000):
        self.buffer = deque(maxlen=size)
 
    def add(self, exp):
        self.buffer.append(exp)
 
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, s2, d = zip(*batch)
        return (
            torch.FloatTensor(s),
            torch.LongTensor(a),
            torch.FloatTensor(r),
            torch.FloatTensor(s2),
            torch.FloatTensor(d)
        )
 
# Training Rainbow DQN on CartPole
env = gym.make("CartPole-v1")
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
 
online_net = RainbowDQN(state_dim, action_dim)
target_net = RainbowDQN(state_dim, action_dim)
target_net.load_state_dict(online_net.state_dict())
 
optimizer = optim.Adam(online_net.parameters(), lr=1e-3)
buffer = ReplayBuffer()
gamma = 0.99
batch_size = 64
episodes = 300
sync_every = 10
 
rewards = []
 
for ep in range(episodes):
    state = env.reset()[0]
    total_reward = 0
    done = False
 
    while not done:
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = online_net(state_tensor)
        action = q_values.argmax().item()
 
        # Occasionally inject noise for exploration
        online_net.reset_noise()
 
        next_state, reward, done, _, _ = env.step(action)
        buffer.add((state, action, reward, next_state, float(done)))
 
        state = next_state
        total_reward += reward
 
        # Train
        if len(buffer.buffer) >= batch_size:
            s, a, r, s2, d = buffer.sample(batch_size)
 
            q_vals = online_net(s).gather(1, a.unsqueeze(1)).squeeze()
            with torch.no_grad():
                next_q = target_net(s2)
                max_next = next_q.argmax(1)
                target_q = r + gamma * (1 - d) * next_q.gather(1, max_next.unsqueeze(1)).squeeze()
 
            loss = nn.MSELoss()(q_vals, target_q)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
 
    if ep % sync_every == 0:
        target_net.load_state_dict(online_net.state_dict())
 
    rewards.append(total_reward)
    print(f"Episode {ep+1}, Reward: {total_reward}")
 
# Plot learning curve
plt.plot(rewards)
plt.title("Rainbow DQN (Noisy + Dueling + Double) on CartPole")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.grid(True)
plt.show()
