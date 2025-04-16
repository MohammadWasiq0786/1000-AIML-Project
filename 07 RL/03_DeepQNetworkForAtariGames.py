"""
Project 243. Deep Q-network for Atari games
Description:
Deep Q-Networks (DQN) use deep neural networks to approximate Q-values in environments with large or continuous state spaces—like Atari games. Instead of a Q-table, DQN learns the value of actions from pixel observations using a convolutional neural network and experience replay. In this project, we’ll train a DQN agent on CartPole-v1 (for simplicity) using PyTorch and OpenAI Gym.

About:
✅ What It Does:
Builds a neural network-based Q-function for CartPole.

Uses experience replay and target network updates.

Learns to balance the pole by choosing optimal left/right moves.

Demonstrates how DQN scales to high-dimensional or continuous environments.
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
 
# Define the DQN network (fully connected for CartPole)
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, output_dim)
        )
 
    def forward(self, x):
        return self.fc(x)
 
# Replay buffer for experience replay
class ReplayBuffer:
    def __init__(self, max_size=10000):
        self.buffer = deque(maxlen=max_size)
 
    def push(self, transition):
        self.buffer.append(transition)
 
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
 
    def __len__(self):
        return len(self.buffer)
 
# Initialize environment and DQN
env = gym.make("CartPole-v1")
obs_dim = env.observation_space.shape[0]
n_actions = env.action_space.n
 
policy_net = DQN(obs_dim, n_actions)
target_net = DQN(obs_dim, n_actions)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()
 
optimizer = optim.Adam(policy_net.parameters(), lr=1e-3)
replay_buffer = ReplayBuffer()
 
# Hyperparameters
gamma = 0.99
epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.995
batch_size = 64
target_update = 10
episodes = 300
 
# Training loop
rewards_all = []
 
for episode in range(episodes):
    state = env.reset()[0]
    total_reward = 0
 
    while True:
        # ε-greedy action selection
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                q_values = policy_net(torch.FloatTensor(state))
                action = q_values.argmax().item()
 
        next_state, reward, done, _, _ = env.step(action)
        replay_buffer.push((state, action, reward, next_state, done))
        state = next_state
        total_reward += reward
 
        # Learn from experience
        if len(replay_buffer) >= batch_size:
            batch = replay_buffer.sample(batch_size)
            states, actions, rewards, next_states, dones = zip(*batch)
 
            states = torch.FloatTensor(states)
            actions = torch.LongTensor(actions).unsqueeze(1)
            rewards = torch.FloatTensor(rewards).unsqueeze(1)
            next_states = torch.FloatTensor(next_states)
            dones = torch.FloatTensor(dones).unsqueeze(1)
 
            # Q(s, a)
            q_values = policy_net(states).gather(1, actions)
 
            # max_a' Q_target(s', a')
            next_q = target_net(next_states).max(1)[0].unsqueeze(1)
            target_q = rewards + gamma * next_q * (1 - dones)
 
            loss = nn.MSELoss()(q_values, target_q)
 
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
 
        if done:
            break
 
    # Decay epsilon
    epsilon = max(epsilon_min, epsilon * epsilon_decay)
    rewards_all.append(total_reward)
 
    # Update target network
    if episode % target_update == 0:
        target_net.load_state_dict(policy_net.state_dict())
 
    print(f"Episode {episode+1}, Reward: {total_reward}, Epsilon: {epsilon:.3f}")
 
# Plot rewards
plt.plot(rewards_all)
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("DQN Training on CartPole")
plt.grid(True)
plt.show()