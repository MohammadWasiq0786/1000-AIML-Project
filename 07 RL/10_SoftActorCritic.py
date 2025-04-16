"""
Project 250. Soft actor-critic implementation
Description:
Soft Actor-Critic (SAC) is a state-of-the-art off-policy reinforcement learning algorithm that optimizes a stochastic policy in an entropy-regularized RL framework. This encourages exploration by maximizing expected reward + entropy, helping the agent explore better and learn more robust policies. SAC is ideal for continuous control tasks like robotics, locomotion, and simulation environments.

About:
✅ What It Does:
Uses entropy regularization to promote better exploration.

Learns a stochastic policy in continuous action space.

Applies twin Q-networks, soft updates, and reparameterization trick.

Great for robotics, manipulation, and continuous navigation.
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
 
# Gaussian policy with reparameterization trick
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU()
        )
        self.mean = nn.Linear(256, action_dim)
        self.log_std = nn.Linear(256, action_dim)
        self.max_action = max_action
 
    def forward(self, state):
        x = self.net(state)
        mean = self.mean(x)
        log_std = torch.clamp(self.log_std(x), -20, 2)
        std = log_std.exp()
 
        # Reparameterization trick
        normal = torch.distributions.Normal(mean, std)
        z = normal.rsample()
        action = torch.tanh(z) * self.max_action
 
        # Compute log_prob
        log_prob = normal.log_prob(z) - torch.log(1 - action.pow(2) + 1e-7)
        log_prob = log_prob.sum(1, keepdim=True)
 
        return action, log_prob
 
    def sample(self, state):
        return self.forward(state)
 
# Critic Q-network
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.q = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 1)
        )
 
    def forward(self, state, action):
        return self.q(torch.cat([state, action], dim=1))
 
# Replay Buffer
class ReplayBuffer:
    def __init__(self, max_size=1_000_000):
        self.buffer = deque(maxlen=max_size)
 
    def add(self, transition):
        self.buffer.append(transition)
 
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, s2, d = zip(*batch)
        return (
            torch.FloatTensor(s),
            torch.FloatTensor(a),
            torch.FloatTensor(r).unsqueeze(1),
            torch.FloatTensor(s2),
            torch.FloatTensor(d).unsqueeze(1)
        )
 
# Environment
env = gym.make("Pendulum-v1")
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
max_action = float(env.action_space.high[0])
 
# Networks
actor = Actor(state_dim, action_dim, max_action)
q1 = QNetwork(state_dim, action_dim)
q2 = QNetwork(state_dim, action_dim)
q1_target = QNetwork(state_dim, action_dim)
q2_target = QNetwork(state_dim, action_dim)
q1_target.load_state_dict(q1.state_dict())
q2_target.load_state_dict(q2.state_dict())
 
# Optimizers
actor_opt = optim.Adam(actor.parameters(), lr=3e-4)
q1_opt = optim.Adam(q1.parameters(), lr=3e-4)
q2_opt = optim.Adam(q2.parameters(), lr=3e-4)
 
# Replay buffer
buffer = ReplayBuffer()
 
# SAC hyperparameters
gamma = 0.99
tau = 0.005
alpha = 0.2  # Entropy coefficient
batch_size = 256
episodes = 150
 
rewards_history = []
 
for episode in range(episodes):
    state = env.reset()[0]
    total_reward = 0
 
    for _ in range(200):
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            action, _ = actor.sample(state_tensor)
            action = action[0].numpy()
 
        next_state, reward, done, _, _ = env.step(action)
        buffer.add((state, action, reward, next_state, float(done)))
        state = next_state
        total_reward += reward
 
        if len(buffer) < batch_size:
            continue
 
        # Sample batch
        s, a, r, s2, d = buffer.sample(batch_size)
 
        # Sample new actions from actor for next state
        with torch.no_grad():
            a2, logp_a2 = actor.sample(s2)
            target_q1 = q1_target(s2, a2)
            target_q2 = q2_target(s2, a2)
            target_q = torch.min(target_q1, target_q2) - alpha * logp_a2
            target_q = r + gamma * (1 - d) * target_q
 
        # Update Q-networks
        q1_loss = nn.MSELoss()(q1(s, a), target_q)
        q2_loss = nn.MSELoss()(q2(s, a), target_q)
        q1_opt.zero_grad()
        q1_loss.backward()
        q1_opt.step()
        q2_opt.zero_grad()
        q2_loss.backward()
        q2_opt.step()
 
        # Update actor
        new_actions, logp = actor.sample(s)
        q1_val = q1(s, new_actions)
        actor_loss = (alpha * logp - q1_val).mean()
        actor_opt.zero_grad()
        actor_loss.backward()
        actor_opt.step()
 
        # Soft update targets
        for t, l in zip(q1_target.parameters(), q1.parameters()):
            t.data.copy_(tau * l.data + (1 - tau) * t.data)
        for t, l in zip(q2_target.parameters(), q2.parameters()):
            t.data.copy_(tau * l.data + (1 - tau) * t.data)
 
    rewards_history.append(total_reward)
    print(f"Episode {episode+1}, Reward: {total_reward:.2f}")
 
# Plot reward curve
plt.plot(rewards_history)
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Soft Actor-Critic (SAC) on Pendulum")
plt.grid(True)
plt.show()