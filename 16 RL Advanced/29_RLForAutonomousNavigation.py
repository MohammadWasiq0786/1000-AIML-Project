"""
Project 629: RL for Autonomous Navigation
Description:
Reinforcement learning (RL) for autonomous navigation involves teaching an agent to navigate a dynamic environment (e.g., a robot, drone, or self-driving car) to reach a target while avoiding obstacles. The agent learns a policy to decide how to act based on sensory inputs, optimizing its path to maximize rewards such as safety, efficiency, or reaching the goal in the shortest time. In this project, we will implement RL for autonomous navigation, using Deep Q-Learning (DQN) to navigate a simple environment.
"""

import gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
 
# 1. Define the Q-network (Deep Q-Network) for autonomous navigation
class QNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_size)  # Output: action probabilities (navigation actions)
 
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)  # Output: Q-values for each action
 
# 2. Define the DQN agent for autonomous navigation
class DQNAgent:
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
 
# 3. Initialize the environment and DQN agent
env = gym.make('LunarLander-v2')  # Example environment for autonomous navigation (landing a spaceship)
model = QNetwork(input_size=env.observation_space.shape[0], output_size=env.action_space.n)
agent = DQNAgent(model)
 
# 4. Train the agent using DQN for autonomous navigation
num_episodes = 1000
for episode in range(num_episodes):
    state = env.reset()
    done = False
    total_reward = 0
    while not done:
        action = agent.select_action(state)
        next_state, reward, done, _, _ = env.step(action)
 
        # Update the agent using DQN
        loss = agent.update(state, action, reward, next_state, done)
        total_reward += reward
        state = next_state
 
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
 
print(f"Total reward after DQN training for Autonomous Navigation: {total_reward}")