"""
Project 265. RL for robotic control
Description:
Reinforcement Learning for Robotic Control focuses on teaching agents (robots) to learn continuous actions such as moving joints, balancing, or walking. These tasks involve complex physics, continuous state/action spaces, and long horizons. We’ll use the popular MuJoCo-inspired Gym environment Pendulum-v1, where the goal is to swing and balance a pendulum upright.

We'll implement a classic Deep Deterministic Policy Gradient (DDPG) agent — an off-policy actor-critic algorithm for continuous control.

About:
✅ What It Does:
Trains an actor-critic architecture for continuous control.

Uses target networks and soft updates for stability.

Adds exploration noise to learn torque control for balancing the pendulum.

Generalizes to robot arms, drones, and autonomous vehicles.
"""


# Install dependencies:
# pip install gym torch numpy matplotlib
 
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import matplotlib.pyplot as plt
 
# Actor network (outputs continuous actions)
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, action_dim), nn.Tanh()
        )
        self.max_action = max_action
 
    def forward(self, x):
        return self.max_action * self.fc(x)
 
# Critic network (Q-function)
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 1)
        )
 
    def forward(self, state, action):
        return self.fc(torch.cat([state, action], dim=1))
 
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
 
# Setup
env = gym.make("Pendulum-v1")
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
 
# Hyperparameters
episodes = 200
batch_size = 128
gamma = 0.99
tau = 0.005
explore_noise = 0.1
 
rewards = []
 
for ep in range(episodes):
    state = env.reset()[0]
    total_reward = 0
 
    for t in range(200):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action = actor(state_tensor).detach().numpy()[0]
        action += explore_noise * np.random.randn(action_dim)
        action = np.clip(action, -max_action, max_action)
 
        next_state, reward, done, _, _ = env.step(action)
        buffer.add((state, action, reward, next_state, float(done)))
        state = next_state
        total_reward += reward
 
        if len(buffer.buffer) < batch_size:
            continue
 
        # Sample batch
        s, a, r, s2, d = buffer.sample(batch_size)
 
        # Critic update
        with torch.no_grad():
            next_a = actor_target(s2)
            target_q = r + gamma * (1 - d) * critic_target(s2, next_a)
        current_q = critic(s, a)
        critic_loss = nn.MSELoss()(current_q, target_q)
 
        critic_opt.zero_grad()
        critic_loss.backward()
        critic_opt.step()
 
        # Actor update
        actor_loss = -critic(s, actor(s)).mean()
        actor_opt.zero_grad()
        actor_loss.backward()
        actor_opt.step()
 
        # Soft update targets
        for target_param, param in zip(actor_target.parameters(), actor.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
        for target_param, param in zip(critic_target.parameters(), critic.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
 
    rewards.append(total_reward)
    print(f"Episode {ep+1}, Total Reward: {total_reward:.2f}")
 
# Plot performance
plt.plot(rewards)
plt.title("DDPG Performance on Pendulum-v1")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.grid(True)
plt.show()