"""
Project 626: RL for Combinatorial Optimization
Description:
Reinforcement learning (RL) for combinatorial optimization focuses on solving combinatorial problems (e.g., traveling salesman problem, knapsack problem) using RL methods. In these problems, the goal is to find the optimal solution from a large set of discrete options. RL can be used to learn policies that guide the search process for optimal solutions in complex, high-dimensional spaces. In this project, we will apply RL to a combinatorial optimization problem, where an agent learns to optimize a combinatorial objective.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
 
# 1. Define a neural network model for RL in combinatorial optimization
class CombinatorialOptimizationModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(CombinatorialOptimizationModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_size)  # Output: action probabilities
 
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return torch.softmax(self.fc3(x), dim=-1)  # Softmax for action probabilities
 
# 2. Define the RL agent for combinatorial optimization
class CombinatorialOptimizationAgent:
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
            action_probs = self.model(torch.tensor(state, dtype=torch.float32))
            return torch.argmax(action_probs).item()  # Select action with the highest probability
 
    def update(self, state, action, reward, next_state, done):
        # Policy gradient update rule
        action_probs = self.model(torch.tensor(state, dtype=torch.float32))
        log_prob = torch.log(action_probs[action])
        loss = -log_prob * reward  # Maximize expected reward (policy gradient)
 
        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
 
        # Decay epsilon (exploration rate)
        if done:
            self.epsilon *= self.epsilon_decay
 
        return loss.item()
 
# 3. Define the combinatorial optimization environment (e.g., Knapsack Problem)
class KnapsackEnv:
    def __init__(self, values, weights, capacity):
        self.values = values
        self.weights = weights
        self.capacity = capacity
        self.num_items = len(values)
        self.state = np.zeros(self.num_items)  # All items are initially not selected
 
    def reset(self):
        # Reset the environment
        self.state = np.zeros(self.num_items)
        return self.state
 
    def step(self, action):
        # Action: 0 = do not pick the item, 1 = pick the item
        if self.state[action] == 1:
            return self.state, 0, True  # Item already picked, no reward, done
 
        self.state[action] = 1  # Mark item as picked
        weight = np.sum(self.state * self.weights)
        if weight > self.capacity:
            self.state[action] = 0  # Undo pick if weight exceeds capacity
            return self.state, -1, False  # Penalty for exceeding capacity
 
        reward = np.sum(self.state * self.values)  # Reward is the total value of picked items
        done = np.all(self.state)  # Done if all items are selected
 
        return self.state, reward, done
 
# 4. Initialize the environment and RL agent
values = np.array([10, 40, 30, 50])  # Item values
weights = np.array([5, 4, 6, 3])  # Item weights
capacity = 10  # Knapsack capacity
env = KnapsackEnv(values, weights, capacity)
 
model = CombinatorialOptimizationModel(input_size=env.num_items, output_size=env.num_items)
agent = CombinatorialOptimizationAgent(model)
 
# 5. Train the agent using policy gradient for combinatorial optimization
num_episodes = 1000
for episode in range(num_episodes):
    state = env.reset()
    done = False
    total_reward = 0
    while not done:
        action = agent.select_action(state)
        next_state, reward, done = env.step(action)
 
        # Update the agent using policy gradient
        loss = agent.update(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward
 
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
 
print(f"Total reward after RL for Combinatorial Optimization training: {total_reward}")