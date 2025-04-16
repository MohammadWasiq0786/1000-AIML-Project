"""
Project 251. Model-based reinforcement learning
Description:
Model-Based Reinforcement Learning (MBRL) builds a predictive model of the environment to plan actions and simulate outcomes. This approach improves sample efficiency by using synthetic experiences (imaginary rollouts) instead of relying entirely on real-world interactions. In this project, we’ll implement a simplified MBRL agent that:

Learns a model of the environment's dynamics (state → next state),

Plans using that model to improve its policy.

We’ll use CartPole-v1 for simplicity and focus on learning the transition dynamics.

About:
✅ What It Does:
Trains a dynamics model to predict next state and reward.

Uses that model to simulate outcomes and choose the best action.

Demonstrates how planning with models can guide decision-making.
"""
# Install dependencies:
# pip install gym torch numpy matplotlib
 
import gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import matplotlib.pyplot as plt
from collections import deque
 
# Dynamics model: predicts next_state and reward given state and action
class DynamicsModel(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim + action_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, state_dim + 1)  # Predict next_state and reward
        )
 
    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        output = self.model(x)
        next_state = output[:, :-1]
        reward = output[:, -1:]
        return next_state, reward
 
# Simple policy: chooses action based on model-predicted rewards
class RandomPolicy:
    def __init__(self, action_space):
        self.action_space = action_space
 
    def get_action(self, _):
        return self.action_space.sample()
 
# Replay buffer
class ReplayBuffer:
    def __init__(self, size=10000):
        self.buffer = deque(maxlen=size)
 
    def add(self, s, a, r, s2):
        self.buffer.append((s, a, r, s2))
 
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, s2 = zip(*batch)
        return (
            torch.FloatTensor(s),
            torch.FloatTensor(a).unsqueeze(1),
            torch.FloatTensor(r).unsqueeze(1),
            torch.FloatTensor(s2)
        )
 
# Initialize environment and models
env = gym.make("CartPole-v1")
state_dim = env.observation_space.shape[0]
action_dim = 1  # Discrete: 0 or 1 (we’ll represent as float)
 
dynamics_model = DynamicsModel(state_dim, action_dim)
optimizer = optim.Adam(dynamics_model.parameters(), lr=1e-3)
 
buffer = ReplayBuffer()
policy = RandomPolicy(env.action_space)
 
# Step 1: Collect real experience with random policy
for _ in range(1000):
    state = env.reset()[0]
    for _ in range(200):
        action = policy.get_action(state)
        next_state, reward, done, _, _ = env.step(action)
        buffer.add(state, float(action), reward, next_state)
        state = next_state
        if done:
            break
 
# Step 2: Train dynamics model
loss_fn = nn.MSELoss()
loss_log = []
 
for epoch in range(100):
    if len(buffer.buffer) < 256:
        continue
 
    s, a, r, s2 = buffer.sample(256)
    pred_next_state, pred_reward = dynamics_model(s, a)
    loss = loss_fn(pred_next_state, s2) + loss_fn(pred_reward, r)
 
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    loss_log.append(loss.item())
 
print("📈 Dynamics model training completed.")
 
# Step 3: Model-based simulation to plan best actions
state = env.reset()[0]
total_reward = 0
 
for _ in range(200):
    state_tensor = torch.FloatTensor(state).unsqueeze(0).repeat(2, 1)  # try both actions: 0 and 1
    action_tensor = torch.FloatTensor([[0], [1]])
    next_states, rewards = dynamics_model(state_tensor, action_tensor)
 
    best_action = torch.argmax(rewards).item()
    state, reward, done, _, _ = env.step(best_action)
    total_reward += reward
 
    env.render()
    if done:
        break
 
env.close()
print(f"🏁 Total reward using learned dynamics model: {total_reward}")

# Plot loss curve for model training
plt.plot(loss_log)
plt.xlabel("Epoch")
plt.ylabel("Model Loss")
plt.title("Dynamics Model Training Loss")
plt.grid(True)
plt.show()