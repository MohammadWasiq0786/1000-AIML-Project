"""
Project 634: RL for Computer Vision Tasks
Description:
Reinforcement learning (RL) for computer vision tasks focuses on applying RL techniques to solve vision-based problems such as image classification, object detection, and segmentation. In this project, we will integrate RL with computer vision tasks, where an agent learns to perform tasks like selecting regions of interest in an image or optimizing object detection models. We'll apply Deep Q-Learning (DQN) to a computer vision task to train an agent to interact with an image-based environment.
"""

import gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchvision import transforms
from PIL import Image
 
# 1. Define the Q-network (Deep Q-Network) for computer vision tasks
class VisionQNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(VisionQNetwork, self).__init__()
        self.conv1 = nn.Conv2d(input_size, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, output_size)  # Output: action probabilities (e.g., select region or class)
 
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = torch.relu(self.fc1(x))
        return self.fc2(x)  # Output: Q-values for each action
 
# 2. Define the RL agent for computer vision tasks
class VisionRLAgent:
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
 
# 3. Define a simple image-based environment (simulated for this task)
class ImageEnv:
    def __init__(self):
        self.state = np.random.rand(3, 32, 32)  # Example: 32x32 image with 3 color channels
        self.num_actions = 4  # Example: 4 possible actions (select region, classify, etc.)
 
    def reset(self):
        self.state = np.random.rand(3, 32, 32)  # Random state (image)
        return self.state
 
    def step(self, action):
        # Simulate the environment step based on the action taken
        # For simplicity, assume reward is based on random chance
        reward = np.random.choice([1, 0], p=[0.7, 0.3])  # 70% chance of positive feedback
        done = False  # In this simple environment, we never "end" the task
        return self.state, reward, done
 
# 4. Initialize the environment and RL agent for computer vision tasks
env = ImageEnv()
model = VisionQNetwork(input_size=3, output_size=env.num_actions)
agent = VisionRLAgent(model)
 
# 5. Train the agent using DQN for computer vision tasks
num_episodes = 1000
for episode in range(num_episodes):
    state = env.reset()
    done = False
    total_reward = 0
    while not done:
        action = agent.select_action(torch.tensor(state, dtype=torch.float32).unsqueeze(0))
        next_state, reward, done = env.step(action)
 
        # Update the agent using DQN
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
 
print(f"Total reward after DQN training for Computer Vision Task: {total_reward}")