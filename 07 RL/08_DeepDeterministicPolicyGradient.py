"""
Project 248. Deep deterministic policy gradient
Description:
DDPG is an actor-critic, model-free reinforcement learning algorithm tailored for continuous action spaces. It combines ideas from DQN (target networks, replay buffer) and deterministic policy gradients to train agents in environments like robotic control, autonomous vehicles, and finance. It uses a separate actor network (for actions) and a critic network (for value estimation), making it ideal for environments like Pendulum-v1 or LunarLanderContinuous-v2.

About:
✅ What It Does:
Learns continuous control using actor-critic architecture.

Uses target networks, experience replay, and soft updates.

Ideal for environments where discrete actions aren't sufficient.
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
 
# Actor network maps state to continuous action
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, action_dim), nn.Tanh()
        )
        self.max_action = max_action
 
    def forward(self, state):
        return self.fc(state) * self.max_action
 
# Critic network maps (state, action) to Q-value
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 1)
        )
 
    def forward(self, state, action):
        return self.fc(torch.cat([state, action], dim=1))
 
# Replay buffer to store transitions
class ReplayBuffer:
    def __init__(self, capacity=1_000_000):
        self.buffer = deque(maxlen=capacity)
 
    def add(self, transition):
        self.buffer.append(transition)
 
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.FloatTensor(states),
            torch.FloatTensor(actions),
            torch.FloatTensor(rewards).unsqueeze(1),
            torch.FloatTensor(next_states),
            torch.FloatTensor(dones).unsqueeze(1)
        )
 
    def __len__(self):
        return len(self.buffer)
 
# Set up environment
env = gym.make("Pendulum-v1")
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
max_action = float(env.action_space.high[0])
 
# Create networks
actor = Actor(state_dim, action_dim, max_action)
critic = Critic(state_dim, action_dim)
target_actor = Actor(state_dim, action_dim, max_action)
target_critic = Critic(state_dim, action_dim)
 
# Copy weights to target networks
target_actor.load_state_dict(actor.state_dict())
target_critic.load_state_dict(critic.state_dict())
 
# Optimizers
actor_optimizer = optim.Adam(actor.parameters(), lr=1e-4)
critic_optimizer = optim.Adam(critic.parameters(), lr=1e-3)
 
# Replay buffer
buffer = ReplayBuffer()
 
# Hyperparameters
gamma = 0.99
tau = 0.005
batch_size = 64
episodes = 150
exploration_noise = 0.1
 
rewards_history = []
 
for episode in range(episodes):
    state = env.reset()[0]
    episode_reward = 0
 
    for _ in range(200):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action = actor(state_tensor).detach().numpy()[0]
        action += exploration_noise * np.random.randn(action_dim)
        action = np.clip(action, -max_action, max_action)
 
        next_state, reward, done, _, _ = env.step(action)
        buffer.add((state, action, reward, next_state, float(done)))
        state = next_state
        episode_reward += reward
 
        if len(buffer) > batch_size:
            # Sample batch
            s, a, r, s2, d = buffer.sample(batch_size)
 
            # Compute target Q value
            with torch.no_grad():
                next_a = target_actor(s2)
                next_q = target_critic(s2, next_a)
                target_q = r + (1 - d) * gamma * next_q
 
            # Update Critic
            q_val = critic(s, a)
            critic_loss = nn.MSELoss()(q_val, target_q)
            critic_optimizer.zero_grad()
            critic_loss.backward()
            critic_optimizer.step()
 
            # Update Actor
            actor_loss = -critic(s, actor(s)).mean()
            actor_optimizer.zero_grad()
            actor_loss.backward()
            actor_optimizer.step()
 
            # Soft update target networks
            for target_param, param in zip(target_critic.parameters(), critic.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
 
            for target_param, param in zip(target_actor.parameters(), actor.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
 
    rewards_history.append(episode_reward)
    print(f"Episode {episode+1}, Reward: {episode_reward:.2f}")
 
# Plot rewards
plt.plot(rewards_history)
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("DDPG on Pendulum")
plt.grid(True)
plt.show()