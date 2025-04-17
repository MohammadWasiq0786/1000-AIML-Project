"""
Project 624: Attention Mechanisms in Reinforcement Learning
Description:
Attention mechanisms are techniques that allow models to focus on specific parts of the input data, selectively weighting the importance of different features. In reinforcement learning, attention mechanisms can help agents prioritize more relevant parts of the environment's state space. This project will implement attention mechanisms in reinforcement learning, where the agent uses attention to decide which aspects of the state are most critical for taking actions.
"""

import gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
 
# 1. Define a simple neural network model with an attention mechanism
class AttentionLayer(nn.Module):
    def __init__(self, input_size):
        super(AttentionLayer, self).__init__()
        self.attention_weights = nn.Parameter(torch.randn(input_size))  # Learnable attention weights
 
    def forward(self, x):
        # Apply attention mechanism: weight the input features
        attention_scores = torch.matmul(x, self.attention_weights)  # Compute attention scores
        attention_probs = torch.softmax(attention_scores, dim=-1)  # Normalize to get probabilities
        return torch.sum(x * attention_probs, dim=-1)  # Weighted sum of input features based on attention
 
class AttentionPolicyNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(AttentionPolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.attention = AttentionLayer(64)  # Attention applied to the hidden layer
        self.fc2 = nn.Linear(64, output_size)  # Output: action probabilities
 
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.attention(x)  # Apply attention to the hidden layer
        return torch.softmax(self.fc2(x), dim=-1)  # Output action probabilities
 
# 2. Define the RL agent with an attention mechanism
class AttentionRLAgent:
    def __init__(self, model, learning_rate=0.001, gamma=0.99, epsilon=1.0, epsilon_decay=0.995):
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.criterion = nn.MSELoss()
 
    def select_action(self, state):
        # Epsilon-greedy action selection
        if np.random.rand() < self.epsilon:
            return np.random.choice(len(state))  # Random action (exploration)
        else:
            q_values = self.model(torch.tensor(state, dtype=torch.float32))
            return torch.argmax(q_values).item()  # Select action with the highest Q-value
 
    def update(self, state, action, reward, next_state, done):
        # Q-learning update rule
        q_values = self.model(torch.tensor(state, dtype=torch.float32))
        next_q_values = self.model(torch.tensor(next_state, dtype=torch.float32))
        target = reward + self.gamma * torch.max(next_q_values) * (1 - done)
        loss = self.criterion(q_values[action], target)  # Compute loss (MSE)
 
        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
 
        # Decay epsilon (exploration rate)
        if done:
            self.epsilon *= self.epsilon_decay
 
        return loss.item()
 
# 3. Initialize the environment and Attention RL agent
env = gym.make('CartPole-v1')
model = AttentionPolicyNetwork(input_size=env.observation_space.shape[0], output_size=env.action_space.n)
agent = AttentionRLAgent(model)
 
# 4. Train the agent using Reinforcement Learning with Attention Mechanism
num_episodes = 1000
for episode in range(num_episodes):
    state = env.reset()
    done = False
    total_reward = 0
    while not done:
        action = agent.select_action(state)
        next_state, reward, done, _, _ = env.step(action)
 
        # Update the agent using the attention-based policy network
        loss = agent.update(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward
 
    if episode % 100 == 0:
        print(f"Episode {episode}, Total Reward: {total_reward}, Loss: {loss:.4f}")
 
# 5. Evaluate the agent after training (no exploration, only exploitation)
state = env.reset()
done = False
total_reward = 0
while not done:
    action = agent.select_action(state)
    next_state, reward, done, _, _ = env.step(action)
    total_reward += reward
    state = next_state
 
print(f"Total reward after RL with Attention training: {total_reward}")