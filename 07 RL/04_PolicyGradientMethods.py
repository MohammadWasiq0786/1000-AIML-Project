"""
Project 244. Policy gradient methods
Description:
Policy Gradient methods learn a parameterized policy directly, instead of estimating value functions. These methods optimize the policy using the gradient of the expected reward with respect to the policy parameters. A popular and simple variant is REINFORCE, which uses Monte Carlo sampling to update the policy. In this project, we'll implement REINFORCE to solve CartPole-v1 using a policy network.

About:
✅ What It Does:
Learns the policy directly using REINFORCE algorithm.

Selects actions based on probability distributions, not Q-values.

Updates policy in the direction that maximizes expected future reward.

Applies to both discrete and continuous action spaces (with different distributions).
"""

# Install dependencies:
# pip install gym torch numpy matplotlib
 
import gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
 
# Define policy network (outputs action probabilities)
class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PolicyNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(),
            nn.Linear(128, output_dim), nn.Softmax(dim=-1)
        )
 
    def forward(self, x):
        return self.fc(x)
 
# Set up environment
env = gym.make("CartPole-v1")
obs_dim = env.observation_space.shape[0]
n_actions = env.action_space.n
 
# Create the policy network
policy_net = PolicyNetwork(obs_dim, n_actions)
optimizer = optim.Adam(policy_net.parameters(), lr=1e-2)
 
# Discount factor
gamma = 0.99
episodes = 500
reward_history = []
 
# REINFORCE training loop
for episode in range(episodes):
    state = env.reset()[0]
    log_probs = []
    rewards = []
 
    # Generate an episode
    while True:
        state_tensor = torch.FloatTensor(state)
        probs = policy_net(state_tensor)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
 
        log_probs.append(dist.log_prob(action))
 
        next_state, reward, done, _, _ = env.step(action.item())
        rewards.append(reward)
 
        state = next_state
        if done:
            break
 
    # Compute discounted returns
    returns = []
    G = 0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    returns = torch.FloatTensor(returns)
 
    # Normalize returns
    returns = (returns - returns.mean()) / (returns.std() + 1e-9)
 
    # Policy loss
    loss = []
    for log_prob, G in zip(log_probs, returns):
        loss.append(-log_prob * G)
    loss = torch.stack(loss).sum()
 
    # Backpropagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
 
    reward_history.append(sum(rewards))
    print(f"Episode {episode+1}, Total Reward: {sum(rewards)}")
 
# Plot reward curve
plt.plot(reward_history)
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("REINFORCE with Policy Gradient on CartPole")
plt.grid(True)
plt.show()