"""
Project 640: RL for Scheduling Problems
Description:
Reinforcement learning (RL) for scheduling problems involves using RL techniques to optimize the allocation of tasks or resources in systems where scheduling is a key factor. This could include optimizing job schedules in factories, computer processes in operating systems, or even employee work shifts. The agent learns to optimize the scheduling policy to maximize efficiency, reduce costs, or meet deadlines. In this project, we will use Q-learning to model a scheduling problem, where the agent learns to allocate tasks to resources in an optimal way.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
 
# 1. Define the Q-network for scheduling optimization
class SchedulingQNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(SchedulingQNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_size)  # Output: Q-values for each action (schedule decision)
 
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)  # Output: Q-values for each action
 
# 2. Define the RL agent for scheduling
class SchedulingRLAgent:
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
 
# 3. Define the scheduling environment (simulated)
class SchedulingEnv:
    def __init__(self, num_jobs, num_resources):
        self.num_jobs = num_jobs
        self.num_resources = num_resources
        self.state = np.random.rand(self.num_jobs)  # Example: job features (e.g., priority, duration)
        self.num_actions = self.num_resources  # Number of available resources for scheduling
 
    def reset(self):
        self.state = np.random.rand(self.num_jobs)  # Random initial state (job features)
        return self.state
 
    def step(self, action):
        # Simulate the environment step based on the action taken (e.g., assigning job to resource)
        # For simplicity, assume reward is based on efficiency of task allocation
        reward = np.random.choice([1, 0], p=[0.7, 0.3])  # 70% chance of positive outcome (efficient scheduling)
        done = False  # In this simple environment, we don't have an "end"
        return self.state, reward, done
 
# 4. Initialize the environment and RL agent for scheduling
env = SchedulingEnv(num_jobs=5, num_resources=3)
model = SchedulingQNetwork(input_size=env.num_jobs, output_size=env.num_resources)
agent = SchedulingRLAgent(model)
 
# 5. Train the agent using Q-learning for scheduling optimization
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
 
print(f"Total reward after Q-learning training for Scheduling: {total_reward}")