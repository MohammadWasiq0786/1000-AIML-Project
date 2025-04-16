"""
Project 246. Proximal policy optimization
Description:
Proximal Policy Optimization (PPO) is a powerful and widely-used policy gradient algorithm that improves training stability by preventing large, destructive policy updates. It uses a clipped surrogate objective to ensure the new policy stays close to the old one. In this project, we'll implement a basic PPO agent using PyTorch and Gym for the CartPole-v1 environment.

About:
✅ What It Does:
Uses an actor-critic network with a clipped loss for stable policy updates.

Applies Generalized Advantage Estimation (GAE) for smoother learning.

Balances exploration vs. exploitation with entropy regularization.

Suitable for complex and high-dimensional environments.
"""

# Install required packages:
# pip install gym torch numpy matplotlib
 
import gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Categorical
import matplotlib.pyplot as plt
 
# Define the PPO Actor-Critic model
class PPOActorCritic(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PPOActorCritic, self).__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU()
        )
        self.actor = nn.Sequential(
            nn.Linear(128, output_dim), nn.Softmax(dim=-1)
        )
        self.critic = nn.Linear(128, 1)
 
    def forward(self, x):
        base = self.shared(x)
        return self.actor(base), self.critic(base)
 
# Collect trajectories
def collect_trajectory(env, model, steps, gamma):
    state = env.reset()[0]
    states, actions, rewards, dones, log_probs, values = [], [], [], [], [], []
 
    for _ in range(steps):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        probs, value = model(state_tensor)
        dist = Categorical(probs)
        action = dist.sample()
 
        next_state, reward, done, _, _ = env.step(action.item())
 
        states.append(state)
        actions.append(action.item())
        rewards.append(reward)
        dones.append(done)
        log_probs.append(dist.log_prob(action))
        values.append(value.item())
 
        state = next_state
        if done:
            state = env.reset()[0]
 
    # Compute returns and advantages
    returns, advantages = [], []
    G, A = 0, 0
    for i in reversed(range(len(rewards))):
        G = rewards[i] + gamma * G * (1 - dones[i])
        delta = rewards[i] + gamma * (values[i+1] if i+1 < len(values) else 0) * (1 - dones[i]) - values[i]
        A = delta + gamma * 0.95 * A * (1 - dones[i])  # GAE with lambda=0.95
        returns.insert(0, G)
        advantages.insert(0, A)
 
    return (
        torch.FloatTensor(states),
        torch.LongTensor(actions),
        torch.FloatTensor(returns),
        torch.FloatTensor(advantages),
        torch.stack(log_probs)
    )
 
# Train PPO
env = gym.make("CartPole-v1")
obs_dim = env.observation_space.shape[0]
n_actions = env.action_space.n
 
model = PPOActorCritic(obs_dim, n_actions)
optimizer = optim.Adam(model.parameters(), lr=3e-4)
 
episodes = 300
batch_steps = 2048
ppo_epochs = 5
clip_epsilon = 0.2
gamma = 0.99
 
episode_rewards = []
state = env.reset()[0]
reward_sum = 0
 
for episode in range(episodes):
    states, actions, returns, advantages, old_log_probs = collect_trajectory(env, model, batch_steps, gamma)
 
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
 
    for _ in range(ppo_epochs):
        probs, values = model(states)
        dist = Categorical(probs)
        entropy = dist.entropy().mean()
        new_log_probs = dist.log_prob(actions)
 
        ratio = (new_log_probs - old_log_probs.detach()).exp()
        clipped = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * advantages
        loss_actor = -torch.min(ratio * advantages, clipped).mean()
 
        loss_critic = nn.MSELoss()(values.squeeze(), returns)
        loss = loss_actor + 0.5 * loss_critic - 0.01 * entropy
 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
 
    # Evaluate one episode to track progress
    state = env.reset()[0]
    done = False
    total_reward = 0
    while not done:
        with torch.no_grad():
            prob, _ = model(torch.FloatTensor(state).unsqueeze(0))
        action = torch.argmax(prob).item()
        state, reward, done, _, _ = env.step(action)
        total_reward += reward
 
    episode_rewards.append(total_reward)
    print(f"Episode {episode+1}, Reward: {total_reward}")
 
# Plot learning curve
plt.plot(episode_rewards)
plt.title("PPO Training on CartPole")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.grid(True)
plt.show()