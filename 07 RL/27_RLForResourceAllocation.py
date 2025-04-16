"""
Project 267. RL for resource allocation
Description:
Resource allocation is a common real-world problem in operations, cloud computing, wireless networks, and more. The idea is to dynamically allocate limited resources (like bandwidth, energy, or CPU time) to competing tasks or users. Reinforcement learning can be used to learn optimal policies that balance fairness, efficiency, and reward.

In this project, we'll simulate a basic multi-job resource allocation problem, where an agent allocates resources to three jobs over time to maximize cumulative reward.

About:
✅ What It Does:
Simulates a multi-job resource demand scenario.

Learns to allocate resources via policy gradients (REINFORCE).

Rewards agent for matching allocation to demand.

Demonstrates a baseline for cloud resource management, load balancing, or task scheduling problems.
"""


import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import random
 
# Simulated environment
class ResourceAllocationEnv:
    def __init__(self):
        self.num_jobs = 3
        self.total_resources = 100
        self.reset()
 
    def reset(self):
        self.job_needs = np.random.randint(10, 50, size=self.num_jobs)
        self.state = self.job_needs.copy()
        return self.state
 
    def step(self, action):
        # Action = how much to allocate to each job (normalized)
        allocation = action / action.sum() * self.total_resources
        rewards = -np.abs(self.job_needs - allocation)  # reward for matching needs
        total_reward = rewards.sum()
        done = True  # single-step decision
        return self.state, total_reward, done
 
# Simple policy network (softmax output for resource share)
class PolicyNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(),
            nn.Linear(64, output_dim), nn.Softmax(dim=-1)
        )
 
    def forward(self, x):
        return self.model(x)
 
# Training using REINFORCE (Policy Gradient)
env = ResourceAllocationEnv()
policy = PolicyNet(input_dim=env.num_jobs, output_dim=env.num_jobs)
optimizer = optim.Adam(policy.parameters(), lr=1e-3)
 
episodes = 1000
reward_log = []
 
for ep in range(episodes):
    state = env.reset()
    state_tensor = torch.FloatTensor(state).unsqueeze(0)
 
    probs = policy(state_tensor)
    dist = torch.distributions.Categorical(probs.squeeze())
    sampled_alloc = dist.sample((env.total_resources,))
    action = torch.bincount(sampled_alloc, minlength=env.num_jobs).float()
 
    _, reward, _ = env.step(action.numpy())
 
    # Policy gradient update
    log_prob = dist.log_prob(sampled_alloc).mean()
    loss = -log_prob * reward
 
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
 
    reward_log.append(reward)
    if (ep + 1) % 100 == 0:
        print(f"Episode {ep+1}, Total Reward: {reward:.2f}")
 
# Plotting rewards
plt.plot(reward_log)
plt.title("RL for Resource Allocation (REINFORCE)")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.grid(True)
plt.show()