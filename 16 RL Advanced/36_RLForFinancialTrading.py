"""
Project 636: RL for Financial Trading
Description:
Reinforcement learning (RL) for financial trading involves training agents to make decisions in financial markets, such as buying, selling, or holding assets, to maximize profits. The goal is to apply RL algorithms to create trading strategies that adapt to market conditions. In this project, we will implement Q-learning for a simple trading environment, where an agent learns to make optimal trades based on market prices.
"""

import gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
 
# 1. Define the Q-network for financial trading
class TradingQNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(TradingQNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_size)  # Output: Q-values for each action (buy, sell, hold)
 
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)  # Output: Q-values for each action
 
# 2. Define the RL agent for financial trading
class TradingRLAgent:
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
 
# 3. Define the financial trading environment (simulated)
class TradingEnv:
    def __init__(self, prices):
        self.prices = prices  # Historical price data
        self.current_step = 0
        self.balance = 1000  # Starting balance
        self.asset = 0  # Amount of asset owned
        self.num_steps = len(prices)
 
    def reset(self):
        self.current_step = 0
        self.balance = 1000
        self.asset = 0
        return np.array([self.balance, self.asset, self.prices[self.current_step]])
 
    def step(self, action):
        current_price = self.prices[self.current_step]
        if action == 0:  # Buy
            self.asset = self.balance / current_price
            self.balance = 0
        elif action == 1:  # Sell
            self.balance = self.asset * current_price
            self.asset = 0
        reward = self.balance + (self.asset * current_price) - 1000  # Reward based on portfolio value
        self.current_step += 1
        done = self.current_step >= self.num_steps
        return np.array([self.balance, self.asset, self.prices[self.current_step]]), reward, done
 
# 4. Initialize the environment and RL agent for financial trading
prices = np.random.uniform(100, 200, 100)  # Example random price data for 100 time steps
env = TradingEnv(prices)
model = TradingQNetwork(input_size=3, output_size=3)  # 3 actions: Buy, Sell, Hold
agent = TradingRLAgent(model)
 
# 5. Train the agent using Q-learning for financial trading
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
 
print(f"Total reward after Q-learning training for Financial Trading: {total_reward}")