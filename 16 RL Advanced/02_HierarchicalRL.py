"""
Project 602: Hierarchical Reinforcement Learning
Description:
Hierarchical reinforcement learning (HRL) involves breaking down a complex task into simpler sub-tasks or hierarchies. This approach is designed to solve tasks more efficiently by training agents to complete sub-tasks and then combine them to achieve the overall goal. In this project, we will implement HRL using a two-level architecture: one level for high-level decisions and another for low-level actions.
"""

import torch
import torch.nn as nn
import numpy as np
import random
import gym
 
# 1. Define a simple environment for HRL (e.g., CartPole)
env = gym.make('CartPole-v1')
 
# 2. High-level policy model (select sub-tasks)
class HighLevelPolicy(nn.Module):
    def __init__(self):
        super(HighLevelPolicy, self).__init__()
        self.fc1 = nn.Linear(4, 64)  # Input: CartPole state (4 dimensions)
        self.fc2 = nn.Linear(64, 2)  # Output: 2 possible high-level tasks (e.g., move left or right)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
 
# 3. Low-level policy model (perform actions based on high-level task)
class LowLevelPolicy(nn.Module):
    def __init__(self):
        super(LowLevelPolicy, self).__init__()
        self.fc1 = nn.Linear(4, 64)  # Input: CartPole state (4 dimensions)
        self.fc2 = nn.Linear(64, 2)  # Output: 2 possible low-level actions (left or right)
 
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
 
# 4. Instantiate models
high_level_policy = HighLevelPolicy()
low_level_policy = LowLevelPolicy()
 
# 5. Define optimizer and loss function
optimizer_high = torch.optim.Adam(high_level_policy.parameters(), lr=0.001)
optimizer_low = torch.optim.Adam(low_level_policy.parameters(), lr=0.001)
criterion = nn.MSELoss()
 
# 6. Training loop (HRL)
num_episodes = 1000
for episode in range(num_episodes):
    state = env.reset()[0]  # Get the initial state from the environment
    done = False
    total_reward = 0
 
    while not done:
        # High-level decision making (select a sub-task)
        high_level_action_probs = high_level_policy(torch.tensor(state, dtype=torch.float32))
        high_level_action = torch.argmax(high_level_action_probs).item()
 
        # Low-level decision making (based on high-level task)
        low_level_action_probs = low_level_policy(torch.tensor(state, dtype=torch.float32))
        low_level_action = torch.argmax(low_level_action_probs).item()
 
        # Perform the action in the environment
        next_state, reward, done, _, _ = env.step(low_level_action)
 
        # Calculate the loss for both high-level and low-level policies
        loss_high = criterion(high_level_action_probs, torch.tensor([high_level_action], dtype=torch.float32))
        loss_low = criterion(low_level_action_probs, torch.tensor([low_level_action], dtype=torch.float32))
 
        # Update the models
        optimizer_high.zero_grad()
        loss_high.backward()
        optimizer_high.step()
 
        optimizer_low.zero_grad()
        loss_low.backward()
        optimizer_low.step()
 
        total_reward += reward
        state = next_state
 
    # Print out results every 100 episodes
    if episode % 100 == 0:
        print(f"Episode {episode}, Total Reward: {total_reward}")