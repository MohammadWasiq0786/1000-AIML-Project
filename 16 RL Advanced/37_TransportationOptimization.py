"""
Project 637: RL for Transportation Optimization
Description:
Reinforcement learning (RL) for transportation optimization involves applying RL algorithms to optimize transportation systems, such as route planning, fleet management, and traffic control. The agent learns to make decisions that minimize travel time, reduce congestion, or optimize fuel efficiency based on real-time data. In this project, we will apply Q-learning to a simple transportation problem, where an agent learns to optimize vehicle routing to minimize travel time and maximize efficiency.
"""

import gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
 
# 1. Define the Q-network for transportation optimization
class TransportationQNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(TransportationQNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_size)  # Output: Q-values for each action (routes)
 
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)  # Output: Q-values for each action (route)
 
# 2. Define the RL agent for transportation optimization
class TransportationRLAgent:
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
            q_values = self.model(state)
            return torch.argmax(q_values).item()  # Select action with the highest Q-value
 
    def update(self, state, action, reward, next_state, done):
        # Q-learning update rule
        q_values = self.model(state)
        next_q_values = self.model(next_state)
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
 
# 3. Define the transportation optimization environment (simulated)
class TransportationEnv:
    def __init__(self):
        self.num_locations = 5  # Number of locations
        self.state = np.random.rand(self.num_locations)  # Example: vehicle's state (location, speed, etc.)
        self.num_actions = self.num_locations  # Number of possible routes (actions)
 
    def reset(self):
        self.state = np.random.rand(self.num_locations)  # Random initial state (e.g., locations, conditions)
        return self.state
 
    def step(self, action):
        # Simulate the environment step based on the action taken (e.g., selecting a route)
        # For simplicity, assume reward is based on the time or distance (lower is better)
        reward = -np.abs(np.sum(self.state) - action)  # Example reward function (minimize travel time/distance)
        done = False  # In this simple environment, we don't have an "end"
        return self.state, reward, done
 
# 4. Initialize the environment and RL agent for transportation optimization
env = TransportationEnv()
model = TransportationQNetwork(input_size=env.num_locations, output_size=env.num_actions)
agent = TransportationRLAgent(model)
 
# 5. Train the agent using Q-learning for transportation optimization
num_episodes = 1000
for episode in range(num_episodes):
    state = env.reset()
    done = False
    total_reward = 0
    while not done:
        action = agent.select_action(torch.tensor(state, dtype=torch.float32).unsqueeze(0))
        next_state, reward, done = env.step(action)
 
        # Update the agent using Q-learning
        loss = agent.update(torch.tensor(state, dtype=torch.float32).unsqueeze(0), action, reward, torch.tensor(next_state, dtype=torch.float32).unsqueeze(0), done)
        total_reward += reward
        state = next_state
 
    if episode % 100 == 0:
        print(f"Episode {episode}, Total Reward: {total_reward}, Loss: {loss:.4f}")
 
# 6. Evaluate the agent after training (no exploration, only exploitation)
state = env.reset()
done = False
total_reward = 0
while not done:
    action = agent.select_action(torch.tensor(state, dtype=torch.float32).unsqueeze(0))
    next_state, reward, done = env.step(action)
    total_reward += reward
    state = next_state
 
print(f"Total reward after Q-learning training for Transportation Optimization: {total_reward}")