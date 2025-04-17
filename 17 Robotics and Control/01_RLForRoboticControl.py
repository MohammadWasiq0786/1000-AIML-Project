"""
Project 641: Reinforcement Learning for Robotic Control
Description:
Reinforcement learning (RL) for robotic control involves teaching robots to perform tasks by interacting with their environment and learning from the feedback received (rewards or penalties). The goal is to train the robot to make optimal decisions, such as adjusting its position, velocity, or other parameters to perform specific tasks like grasping, moving, or navigating. In this project, we will apply RL techniques to train a robot to perform control tasks, using Q-learning or Deep Q-Networks (DQN) for learning from interactions with the environment.
"""

import gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
 
# 1. Define the Q-network for robotic control
class RoboticControlQNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(RoboticControlQNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_size)  # Output: Q-values for each action (control command)
 
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)  # Output: Q-values for each action
 
# 2. Define the RL agent for robotic control
class RoboticControlRLAgent:
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
 
# 3. Initialize the environment and RL agent for robotic control
env = gym.make('FetchReach-v1')  # Example robotic control task (Fetch robot reaching task)
model = RoboticControlQNetwork(input_size=env.observation_space.shape[0], output_size=env.action_space.shape[0])
agent = RoboticControlRLAgent(model)
 
# 4. Train the agent using DQN for robotic control
num_episodes = 1000
for episode in range(num_episodes):
    state = env.reset()
    done = False
    total_reward = 0
    while not done:
        action = agent.select_action(torch.tensor(state, dtype=torch.float32).unsqueeze(0))
        next_state, reward, done, _, _ = env.step(action)
 
        # Update the agent using DQN
        loss = agent.update(torch.tensor(state, dtype=torch.float32).unsqueeze(0), action, reward, torch.tensor(next_state, dtype=torch.float32).unsqueeze(0), done)
        total_reward += reward
        state = next_state
 
    if episode % 100 == 0:
        print(f"Episode {episode}, Total Reward: {total_reward}, Loss: {loss:.4f}")
 
# 5. Evaluate the agent after training (no exploration, only exploitation)
state = env.reset()
done = False
total_reward = 0
while not done:
    action = agent.select_action(torch.tensor(state, dtype=torch.float32).unsqueeze(0))
    next_state, reward, done, _, _ = env.step(action)
    total_reward += reward
    state = next_state
 
print(f"Total reward after DQN training for Robotic Control: {total_reward}")