"""
Project 631: RL for Recommendation Systems
Description:
Reinforcement learning (RL) for recommendation systems involves using RL techniques to optimize the recommendation process by dynamically adjusting recommendations based on user feedback and interactions. In traditional recommendation systems, recommendations are made based on pre-collected data, but RL allows the system to learn and adapt in real-time, improving recommendations by considering long-term user satisfaction. In this project, we will use Q-learning to model a recommendation system that learns to recommend items based on user interactions and rewards.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
 
# 1. Define the Q-network for RL-based recommendation system
class QNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_size)  # Output: Q-values for each action (recommendation)
 
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)  # Output: Q-values for each recommendation
 
# 2. Define the RL agent for recommendation system
class RLRecommendationAgent:
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
 
# 3. Define the recommendation environment (simulated)
class RecommendationEnv:
    def __init__(self, num_users, num_items):
        self.num_users = num_users
        self.num_items = num_items
        self.state = np.zeros(self.num_users)  # Simulate user interactions
 
    def reset(self):
        self.state = np.zeros(self.num_users)
        return self.state
 
    def step(self, action):
        # Action: Recommend an item to the user
        # For simplicity, assume reward is based on a random chance of satisfaction
        reward = np.random.choice([1, 0], p=[0.7, 0.3])  # 70% chance of positive feedback
        done = False  # In this simple environment, we never "end" the recommendation task
        return self.state, reward, done
 
# 4. Initialize the environment and RL agent for recommendation system
num_users = 5
num_items = 10
env = RecommendationEnv(num_users=num_users, num_items=num_items)
model = QNetwork(input_size=num_users, output_size=num_items)
agent = RLRecommendationAgent(model)
 
# 5. Train the agent using Q-learning for recommendation system
num_episodes = 1000
for episode in range(num_episodes):
    state = env.reset()
    done = False
    total_reward = 0
    while not done:
        action = agent.select_action(state)
        next_state, reward, done = env.step(action)
 
        # Update the agent using Q-learning
        loss = agent.update(state, action, reward, next_state, done)
        total_reward += reward
        state = next_state
 
    if episode % 100 == 0:
        print(f"Episode {episode}, Total Reward: {total_reward}, Loss: {loss:.4f}")
 
# 6. Evaluate the agent after training (no exploration, only exploitation)
state = env.reset()
done = False
total_reward = 0
while not done:
    action = agent.select_action(state)
    next_state, reward, done = env.step(action)
    total_reward += reward
    state = next_state
 
print(f"Total reward after Q-learning training for Recommendation System: {total_reward}")