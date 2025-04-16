"""
Project 245. Actor-critic implementation
Description:
The Actor-Critic method combines the strengths of policy-based (actor) and value-based (critic) approaches.

The Actor decides which action to take.

The Critic evaluates how good the action was (i.e., estimates the value function).

This setup reduces variance in policy gradients (like REINFORCE) and enables more stable learning. In this project, we’ll implement a basic Actor-Critic agent for CartPole-v1.

About:
✅ What It Does:
Learns both the policy (actor) and value function (critic) simultaneously.

Reduces the variance of policy updates by guiding them with critic estimates.

Performs more stable and efficient learning than basic REINFORCE.
"""
# Install required packages:
# pip install gym torch numpy matplotlib
 
import gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
 
# Define Actor-Critic network (shared base)
class ActorCritic(nn.Module):
    def __init__(self, obs_dim, n_actions):
        super(ActorCritic, self).__init__()
        self.common = nn.Sequential(
            nn.Linear(obs_dim, 128), nn.ReLU()
        )
        self.actor = nn.Sequential(
            nn.Linear(128, n_actions),
            nn.Softmax(dim=-1)
        )
        self.critic = nn.Linear(128, 1)
 
    def forward(self, x):
        base = self.common(x)
        probs = self.actor(base)
        value = self.critic(base)
        return probs, value
 
# Set up environment
env = gym.make("CartPole-v1")
obs_dim = env.observation_space.shape[0]
n_actions = env.action_space.n
 
# Initialize model and optimizer
model = ActorCritic(obs_dim, n_actions)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
 
# Hyperparameters
gamma = 0.99
episodes = 500
reward_history = []
 
# Training loop
for episode in range(episodes):
    state = env.reset()[0]
    log_probs = []
    values = []
    rewards = []
 
    while True:
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        probs, value = model(state_tensor)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
 
        log_probs.append(dist.log_prob(action))
        values.append(value.squeeze())
        
        next_state, reward, done, _, _ = env.step(action.item())
        rewards.append(reward)
        state = next_state
 
        if done:
            break
 
    # Compute returns and advantages
    returns = []
    G = 0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    returns = torch.FloatTensor(returns)
    values = torch.stack(values)
    log_probs = torch.stack(log_probs)
 
    advantage = returns - values.detach()
 
    # Actor loss: -log_prob * advantage
    actor_loss = -(log_probs * advantage).sum()
 
    # Critic loss: MSE between predicted and actual returns
    critic_loss = nn.MSELoss()(values, returns)
 
    # Total loss
    loss = actor_loss + critic_loss
 
    # Backprop and update
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
 
    total_reward = sum(rewards)
    reward_history.append(total_reward)
    print(f"Episode {episode+1}, Reward: {total_reward:.2f}")
 
# Plot total rewards
plt.plot(reward_history)
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Actor-Critic on CartPole")
plt.grid(True)
plt.show()