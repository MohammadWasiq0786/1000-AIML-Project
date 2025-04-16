"""
Project 249. Twin delayed DDPG
Description:
TD3 improves upon DDPG by addressing overestimation bias in value function estimates. It introduces three core innovations:

Twin Q-networks to mitigate overestimation.

Delayed policy updates to improve stability.

Target policy smoothing to reduce variance in target Q-values.

This project implements TD3 on a continuous control task like Pendulum-v1 using PyTorch.

About:
✅ What It Does:
Uses two critics to prevent Q-value overestimation.

Applies target policy smoothing for stable training.

Delays actor updates to allow critic convergence.

Ideal for robotic arm control, drone navigation, and more.
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
 
# Actor network
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
        return self.model(state) * self.max_action
 
# Twin Critic networks (Q1 and Q2)
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.q1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 1)
        )
        self.q2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 1)
        )
 
    def forward(self, state, action):
        sa = torch.cat([state, action], dim=1)
        return self.q1(sa), self.q2(sa)
 
    def q1_only(self, state, action):
        return self.q1(torch.cat([state, action], dim=1))
 
# Replay buffer
class ReplayBuffer:
    def __init__(self, size=1_000_000):
        self.buffer = deque(maxlen=size)
 
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
 
# Setup
env = gym.make("Pendulum-v1")
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
max_action = float(env.action_space.high[0])
 
# Networks
actor = Actor(state_dim, action_dim, max_action)
actor_target = Actor(state_dim, action_dim, max_action)
actor_target.load_state_dict(actor.state_dict())
 
critic = Critic(state_dim, action_dim)
critic_target = Critic(state_dim, action_dim)
critic_target.load_state_dict(critic.state_dict())
 
# Optimizers
actor_opt = optim.Adam(actor.parameters(), lr=1e-4)
critic_opt = optim.Adam(critic.parameters(), lr=1e-3)
 
# Hyperparameters
buffer = ReplayBuffer()
gamma = 0.99
tau = 0.005
policy_noise = 0.2
noise_clip = 0.5
policy_delay = 2
batch_size = 100
episodes = 150
exploration_noise = 0.1
 
reward_log = []
 
# Training loop
total_steps = 0
for ep in range(episodes):
    state = env.reset()[0]
    ep_reward = 0
    done = False
 
    while not done:
        total_steps += 1
 
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action = actor(state_tensor).detach().numpy()[0]
        action += exploration_noise * np.random.randn(action_dim)
        action = np.clip(action, -max_action, max_action)
 
        next_state, reward, done, _, _ = env.step(action)
        buffer.add((state, action, reward, next_state, float(done)))
        state = next_state
        ep_reward += reward
 
        # Learn if enough samples
        if len(buffer) > batch_size:
            s, a, r, s2, d = buffer.sample(batch_size)
 
            # Add clipped noise to next action (target policy smoothing)
            noise = (torch.randn_like(a) * policy_noise).clamp(-noise_clip, noise_clip)
            next_action = (actor_target(s2) + noise).clamp(-max_action, max_action)
 
            # Compute target Q-values using both target critics
            target_q1, target_q2 = critic_target(s2, next_action)
            target_q = r + gamma * (1 - d) * torch.min(target_q1, target_q2)
 
            # Current Q estimates
            current_q1, current_q2 = critic(s, a)
 
            # Critic loss
            critic_loss = nn.MSELoss()(current_q1, target_q) + nn.MSELoss()(current_q2, target_q)
            critic_opt.zero_grad()
            critic_loss.backward()
            critic_opt.step()
 
            # Delayed policy update
            if total_steps % policy_delay == 0:
                actor_loss = -critic.q1_only(s, actor(s)).mean()
                actor_opt.zero_grad()
                actor_loss.backward()
                actor_opt.step()
 
                # Update targets
                for t_param, param in zip(actor_target.parameters(), actor.parameters()):
                    t_param.data.copy_(tau * param.data + (1 - tau) * t_param.data)
 
                for t_param, param in zip(critic_target.parameters(), critic.parameters()):
                    t_param.data.copy_(tau * param.data + (1 - tau) * t_param.data)
 
    reward_log.append(ep_reward)
    print(f"Episode {ep+1}, Reward: {ep_reward:.2f}")
 
# Plotting
plt.plot(reward_log)
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("TD3 on Pendulum-v1")
plt.grid(True)
plt.show()