"""
Project 606: Imitation Learning from Demonstrations
Description:
Imitation learning (or learning from demonstrations) involves training an agent to mimic the behavior of an expert agent by observing its actions. Instead of learning solely from rewards, the agent learns by trying to replicate the actions of a demonstrator in a given environment. In this project, we will implement imitation learning where an agent learns from expert demonstrations.
"""

import numpy as np
import gym
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
 
# 1. Define the behavioral cloning model (simple neural network)
class BehavioralCloningModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(BehavioralCloningModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_size)  # Output: Action space size
 
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)
 
# 2. Define the imitation learning agent
class ImitationLearningAgent:
    def __init__(self, model, learning_rate=0.001):
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()
 
    def train(self, states, actions):
        self.model.train()
        self.optimizer.zero_grad()
        # Forward pass
        predictions = self.model(states)
        # Compute loss
        loss = self.criterion(predictions, actions)
        # Backpropagation
        loss.backward()
        self.optimizer.step()
        return loss.item()
 
    def select_action(self, state):
        self.model.eval()
        with torch.no_grad():
            action_probs = self.model(state)
        action = torch.argmax(action_probs, dim=-1)
        return action.item()
 
# 3. Load the environment and the expert's demonstration
env = gym.make('CartPole-v1')
 
# Expert agent demonstrations (for simplicity, we'll use random actions)
expert_data = []
for episode in range(100):  # Collect data from 100 episodes
    state = env.reset()
    done = False
    while not done:
        action = env.action_space.sample()  # Expert's action (random in this case)
        next_state, reward, done, _ = env.step(action)
        expert_data.append((state, action))
        state = next_state
 
# Convert expert data to arrays
expert_states = torch.tensor([data[0] for data in expert_data], dtype=torch.float32)
expert_actions = torch.tensor([data[1] for data in expert_data], dtype=torch.long)
 
# 4. Initialize the behavioral cloning model
model = BehavioralCloningModel(input_size=env.observation_space.shape[0], output_size=env.action_space.n)
agent = ImitationLearningAgent(model)
 
# 5. Train the agent on expert data (behavioral cloning)
num_epochs = 50
for epoch in range(num_epochs):
    loss = agent.train(expert_states, expert_actions)
    print(f"Epoch {epoch + 1}, Loss: {loss}")
 
# 6. Evaluate the trained agent (imitation)
state = env.reset()
done = False
total_reward = 0
while not done:
    action = agent.select_action(torch.tensor(state, dtype=torch.float32))
    next_state, reward, done, _ = env.step(action)
    total_reward += reward
    state = next_state
 
print(f"Total reward after imitation learning: {total_reward}")