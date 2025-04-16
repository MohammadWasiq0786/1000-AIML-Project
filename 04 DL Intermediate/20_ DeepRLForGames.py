"""
Project 140. Deep reinforcement learning for games
Description:
Deep Reinforcement Learning (DRL) combines reinforcement learning with deep neural networks to learn optimal actions from raw states like pixels or numerical observations. In this project, we implement a Deep Q-Network (DQN) to play simple games (e.g., CartPole) using the OpenAI Gym environment.

About:
🧠 What This Project Demonstrates:
Implements a Deep Q-Network (DQN) for decision-making in a game

Uses experience replay and target networks for stability

Trains an agent to balance a pole in CartPole-v1 from scratch
"""

# Install if not already: pip install gym torch numpy matplotlib
 
import gym
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque
 
# Define Q-Network
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )
    def forward(self, x):
        return self.model(x)
 
# Replay Buffer
class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return map(np.array, (state, action, reward, next_state, done))
    def __len__(self):
        return len(self.buffer)
 
# Environment
env = gym.make("CartPole-v1")
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
 
# Initialize
q_network = QNetwork(state_dim, action_dim)
target_network = QNetwork(state_dim, action_dim)
target_network.load_state_dict(q_network.state_dict())
 
optimizer = optim.Adam(q_network.parameters(), lr=1e-3)
replay_buffer = ReplayBuffer()
 
# Hyperparameters
episodes = 100
batch_size = 64
gamma = 0.99
epsilon = 1.0
epsilon_decay = 0.995
min_epsilon = 0.01
update_target_every = 10
 
# Training Loop
for episode in range(episodes):
    state = env.reset()[0]
    total_reward = 0
    done = False
 
    while not done:
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                action = q_network(state_tensor).argmax().item()
 
        next_state, reward, done, _, _ = env.step(action)
        replay_buffer.push(state, action, reward, next_state, done)
 
        state = next_state
        total_reward += reward
 
        # Sample and train
        if len(replay_buffer) > batch_size:
            s, a, r, s2, d = replay_buffer.sample(batch_size)
            s = torch.FloatTensor(s)
            a = torch.LongTensor(a)
            r = torch.FloatTensor(r)
            s2 = torch.FloatTensor(s2)
            d = torch.FloatTensor(d)
 
            q_values = q_network(s)
            q_value = q_values.gather(1, a.unsqueeze(1)).squeeze(1)
 
            with torch.no_grad():
                target_q = target_network(s2).max(1)[0]
                expected_q = r + gamma * target_q * (1 - d)
 
            loss = nn.MSELoss()(q_value, expected_q)
 
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
 
    # Update epsilon and target network
    epsilon = max(min_epsilon, epsilon * epsilon_decay)
    if episode % update_target_every == 0:
        target_network.load_state_dict(q_network.state_dict())
 
    print(f"🎮 Episode {episode+1} - Total Reward: {total_reward:.0f}, Epsilon: {epsilon:.3f}")