"""
Project 638: RL for Energy Management
Description:
Reinforcement learning (RL) for energy management involves applying RL algorithms to optimize the generation, distribution, and consumption of energy in systems such as smart grids, renewable energy systems, and building energy management. In this project, we will train an RL agent to make energy management decisions, such as when to store energy, when to consume it, or when to distribute it to minimize costs or maximize efficiency. The agent will learn policies that can handle uncertainties in energy generation and consumption.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
 
# 1. Define the Q-network for energy management optimization
class EnergyQNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(EnergyQNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_size)  # Output: Q-values for each action (energy management decision)
 
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)  # Output: Q-values for each action
 
# 2. Define the RL agent for energy management
class EnergyRLAgent:
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
 
# 3. Define the energy management environment (simulated)
class EnergyManagementEnv:
    def __init__(self):
        self.state = np.random.rand(3)  # Example: 3-dimensional state (e.g., energy storage, demand, price)
        self.num_actions = 3  # Example: 3 possible actions (store energy, consume energy, distribute energy)
 
    def reset(self):
        self.state = np.random.rand(3)  # Random initial state (energy storage, demand, price)
        return self.state
 
    def step(self, action):
        # Simulate the environment step based on the action taken
        # For simplicity, assume reward is based on cost savings or efficiency
        reward = np.random.choice([1, 0], p=[0.7, 0.3])  # 70% chance of positive outcome
        done = False  # In this simple environment, we don't have an "end"
        return self.state, reward, done
 
# 4. Initialize the environment and RL agent for energy management
env = EnergyManagementEnv()
model = EnergyQNetwork(input_size=env.state.shape[0], output_size=env.num_actions)
agent = EnergyRLAgent(model)
 
# 5. Train the agent using Q-learning for energy management
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
 
print(f"Total reward after Q-learning training for Energy Management: {total_reward}")