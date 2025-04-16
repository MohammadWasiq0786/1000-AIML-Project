"""
Project 259. Curiosity-driven exploration
Description:
Curiosity-Driven Exploration helps an agent explore environments without external rewards by generating intrinsic motivation. The idea is to reward the agent for visiting novel or unpredictable states — encouraging it to explore the unknown. This is useful in sparse reward settings (like Montezuma’s Revenge) where extrinsic signals are rare.

In this project, we'll implement a simple curiosity module that rewards the agent for visiting new or surprising states, using prediction error from a learned model.

About:
✅ What It Does:
Builds a forward dynamics model to predict next state from (state, action).

Calculates intrinsic reward as the prediction error (surprise).

Updates Q-values using intrinsic reward — no external reward is needed!

Demonstrates how an agent can explore effectively on its own.
"""


# Install dependencies:
# pip install gym torch numpy matplotlib
 
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from collections import defaultdict
 
# FrozenLake environment (no reward modification)
env = gym.make("FrozenLake-v1", is_slippery=False)
n_states = env.observation_space.n
n_actions = env.action_space.n
 
# Curiosity module: predicts next state from current state and action
class CuriosityModel(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim + action_dim, 32),
            nn.ReLU(),
            nn.Linear(32, state_dim)
        )
 
    def forward(self, state_onehot, action_onehot):
        x = torch.cat([state_onehot, action_onehot], dim=1)
        return self.model(x)
 
# One-hot encoders
def one_hot(n, i):
    vec = np.zeros(n)
    vec[i] = 1
    return vec
 
# Initialize curiosity model
curiosity_model = CuriosityModel(state_dim=n_states, action_dim=n_actions)
curiosity_optimizer = optim.Adam(curiosity_model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()
 
# Q-table for agent
Q = np.zeros((n_states, n_actions))
 
# Parameters
episodes = 500
alpha = 0.1
gamma = 0.99
epsilon = 0.1
 
intrinsic_rewards = []
 
for ep in range(episodes):
    state = env.reset()[0]
    done = False
    ep_intrinsic = 0
 
    while not done:
        # Epsilon-greedy action
        if np.random.rand() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q[state])
 
        next_state, _, done, _, _ = env.step(action)
 
        # Create one-hot encodings
        s_oh = torch.FloatTensor([one_hot(n_states, state)])
        a_oh = torch.FloatTensor([one_hot(n_actions, action)])
        ns_oh = torch.FloatTensor([one_hot(n_states, next_state)])
 
        # Predict next state
        pred_ns = curiosity_model(s_oh, a_oh)
 
        # Curiosity = prediction error
        curiosity = loss_fn(pred_ns, ns_oh).item()
        ep_intrinsic += curiosity
 
        # Q-update using intrinsic reward
        Q[state][action] += alpha * (curiosity + gamma * np.max(Q[next_state]) - Q[state][action])
 
        # Train curiosity model
        loss = loss_fn(pred_ns, ns_oh)
        curiosity_optimizer.zero_grad()
        loss.backward()
        curiosity_optimizer.step()
 
        state = next_state
 
    intrinsic_rewards.append(ep_intrinsic)
    print(f"Episode {ep+1}, Intrinsic Reward: {ep_intrinsic:.4f}")
 
# Plot intrinsic reward over time
plt.plot(intrinsic_rewards)
plt.xlabel("Episode")
plt.ylabel("Total Intrinsic Reward")
plt.title("Curiosity-Driven Exploration in FrozenLake")
plt.grid(True)
plt.show()