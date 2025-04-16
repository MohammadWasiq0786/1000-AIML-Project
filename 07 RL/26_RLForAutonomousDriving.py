"""
Project 266. RL for autonomous driving
Description:
Reinforcement Learning for Autonomous Driving focuses on training an agent to navigate safely by observing its environment, avoiding obstacles, and following traffic rules. In this simplified simulation, we’ll train an RL agent to stay on a track using OpenAI Gym’s CarRacing-v2 environment, which simulates a top-down view of a self-driving car.

We'll use Proximal Policy Optimization (PPO) — a stable and popular on-policy algorithm suitable for environments with continuous action spaces and visual inputs.

About:
✅ What It Does:
Uses PPO to handle continuous action control (steer, gas, brake).

Processes raw visual inputs via a CNN encoder.

Encourages stable driving behavior using policy gradient updates.

Mimics foundational structure behind Autonomous Driving RL simulators.
"""


# Install dependencies:
# pip install gym[box2d] torch torchvision numpy matplotlib
 
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import matplotlib.pyplot as plt
from torchvision import transforms
 
# Preprocessing: resize and grayscale frames
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((64, 64)),
    transforms.Grayscale(),
    transforms.ToTensor()
])
 
# PPO Policy and Value Network
class ActorCritic(nn.Module):
    def __init__(self, input_shape, action_dim):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 8, stride=4), nn.ReLU(),
            nn.Conv2d(16, 32, 4, stride=2), nn.ReLU()
        )
        conv_out_size = self._get_conv_out(input_shape)
 
        self.policy = nn.Sequential(
            nn.Linear(conv_out_size, 256), nn.ReLU(),
            nn.Linear(256, action_dim), nn.Tanh()  # continuous actions: steer, gas, brake
        )
 
        self.value = nn.Sequential(
            nn.Linear(conv_out_size, 256), nn.ReLU(),
            nn.Linear(256, 1)
        )
 
    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))
 
    def forward(self, x):
        features = self.conv(x).view(x.size(0), -1)
        return self.policy(features), self.value(features)
 
# Normalize observation
def preprocess(obs):
    img = transform(obs).unsqueeze(0)  # shape: (1, 1, 64, 64)
    return img
 
# Environment
env = gym.make("CarRacing-v2", continuous=True, domain_randomize=False)
obs_shape = (1, 64, 64)
action_dim = 3  # [steer, gas, brake]
 
model = ActorCritic(obs_shape, action_dim)
optimizer = optim.Adam(model.parameters(), lr=3e-4)
gamma = 0.99
 
# Training loop
rewards = []
episodes = 100
 
for ep in range(episodes):
    obs = env.reset()[0]
    state = preprocess(obs)
    done = False
    total_reward = 0
    log_probs = []
    values = []
    rewards_ep = []
 
    while not done:
        with torch.no_grad():
            action_mean, value = model(state)
        action = action_mean.squeeze().numpy()
        noise = np.random.normal(0, 0.1, size=3)
        action = np.clip(action + noise, -1, 1)
 
        next_obs, reward, done, _, _ = env.step(action)
        next_state = preprocess(next_obs)
 
        # Compute log_prob assuming Gaussian distribution
        dist = torch.distributions.Normal(action_mean, 0.1)
        log_prob = dist.log_prob(torch.FloatTensor(action)).sum()
 
        log_probs.append(log_prob)
        values.append(value.squeeze())
        rewards_ep.append(reward)
 
        state = next_state
        total_reward += reward
 
    # Compute returns and advantage
    returns = []
    G = 0
    for r in reversed(rewards_ep):
        G = r + gamma * G
        returns.insert(0, G)
 
    returns = torch.FloatTensor(returns)
    values = torch.stack(values)
    log_probs = torch.stack(log_probs)
    advantage = returns - values.detach()
 
    # PPO loss
    actor_loss = -(log_probs * advantage).mean()
    critic_loss = nn.MSELoss()(values, returns)
    loss = actor_loss + 0.5 * critic_loss
 
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
 
    rewards.append(total_reward)
    print(f"Episode {ep+1}, Reward: {total_reward:.2f}")
 
# Plot reward trend
plt.plot(rewards)
plt.title("PPO Agent on CarRacing-v2")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.grid(True)
plt.show()