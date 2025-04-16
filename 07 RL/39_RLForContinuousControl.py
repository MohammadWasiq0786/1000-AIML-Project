"""
Project 279. RL for continuous control
Description:
Many real-world tasks require continuous actions — think of robotic arms, drones, or self-driving cars. Unlike discrete actions (e.g., “move left”), continuous control involves actions like “rotate joint by 0.03 radians.” Here, we apply Deep Deterministic Policy Gradient (DDPG) — a classic actor-critic algorithm designed for continuous action spaces.

In this project, we'll use the MountainCarContinuous-v0 environment from OpenAI Gym, where the goal is to apply the right force to push a car up a hill.

About:
✅ What It Does:
Uses DDPG to learn continuous force control.

Trains actor-critic networks with soft updates and noise for exploration.

Learns to swing the car back and forth until it reaches the goal.
"""

import gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import matplotlib.pyplot as plt
 
# Actor network: maps state to continuous action
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, action_dim), nn.Tanh()
        )
        self.max_action = max_action
 
    def forward(self, state):
        return self.max_action * self.model(state)
 
# Critic network: maps (state, action) to Q-value
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 1)
        )
 
    def forward(self, state, action):
        return self.model(torch.cat([state, action], dim=1))
 
# Replay buffer
class ReplayBuffer:
    def __init__(self, size=100000):
        self.buffer = deque(maxlen=size)
 
    def add(self, experience):
        self.buffer.append(experience)
 
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
 
# Setup environment
env = gym.make("MountainCarContinuous-v0")
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
max_action = float(env.action_space.high[0])
 
actor = Actor(state_dim, action_dim, max_action)
actor_target = Actor(state_dim, action_dim, max_action)
actor_target.load_state_dict(actor.state_dict())
 
critic = Critic(state_dim, action_dim)
critic_target = Critic(state_dim, action_dim)
critic_target.load_state_dict(critic.state_dict())
 
actor_opt = optim.Adam(actor.parameters(), lr=1e-3)
critic_opt = optim.Adam(critic.parameters(), lr=1e-3)
 
buffer = ReplayBuffer()
gamma = 0.99
tau = 0.005
batch_size = 64
rewards = []
 
# Training loop
for ep in range(150):
    state = env.reset()[0]
    total_reward = 0
 
    for _ in range(1000):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action = actor(state_tensor).detach().numpy()[0]
        action += np.random.normal(0, 0.1, size=action_dim)  # exploration noise
        action = np.clip(action, -max_action, max_action)
 
        next_state, reward, done, _, _ = env.step(action)
        buffer.add((state, action, reward, next_state, float(done)))
        state = next_state
        total_reward += reward
 
        if len(buffer.buffer) > batch_size:
            s, a, r, s2, d = buffer.sample(batch_size)
            target_q = critic_target(s2, actor_target(s2)).detach()
            q_target = r + gamma * (1 - d) * target_q
 
            q_current = critic(s, a)
            critic_loss = nn.MSELoss()(q_current, q_target)
            critic_opt.zero_grad()
            critic_loss.backward()
            critic_opt.step()
 
            # Actor update
            actor_loss = -critic(s, actor(s)).mean()
            actor_opt.zero_grad()
            actor_loss.backward()
            actor_opt.step()
 
            # Soft updates
            for t, s in zip(actor_target.parameters(), actor.parameters()):
                t.data.copy_(tau * s.data + (1 - tau) * t.data)
            for t, s in zip(critic_target.parameters(), critic.parameters()):
                t.data.copy_(tau * s.data + (1 - tau) * t.data)
 
        if done:
            break
 
    rewards.append(total_reward)
    print(f"Episode {ep+1}, Total Reward: {total_reward:.2f}")
 
# Plot reward trend
plt.plot(rewards)
plt.title("DDPG – Continuous Control (MountainCar)")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.grid(True)
plt.show()