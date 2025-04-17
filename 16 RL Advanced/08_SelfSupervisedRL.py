"""
Project 608: Self-supervised Reinforcement Learning
Description:
Self-supervised reinforcement learning is a technique where the agent learns to predict parts of the environment’s state or action space as a supervisory signal, without explicit human-provided rewards or labeled data. Instead of relying entirely on external rewards, the agent can learn useful representations and skills from the environment through self-generated tasks, such as predicting the next state or the consequences of actions. In this project, we will implement self-supervised learning for reinforcement learning tasks.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gym
 
# 1. Define a simple neural network model for self-supervised learning
class PredictiveModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(PredictiveModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_size)  # Predict the next state or future state
 
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)
 
# 2. Define the self-supervised reinforcement learning agent
class SelfSupervisedRLAgent:
    def __init__(self, model, learning_rate=0.001):
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
 
    def train(self, states, next_states):
        self.model.train()
        self.optimizer.zero_grad()
 
        # Predict the next state using the current state
        predicted_next_state = self.model(states)
 
        # Calculate the loss between the predicted next state and the actual next state
        loss = self.criterion(predicted_next_state, next_states)
        loss.backward()
        self.optimizer.step()
        return loss.item()
 
    def predict_next_state(self, state):
        self.model.eval()
        with torch.no_grad():
            return self.model(state)
 
# 3. Load the environment
env = gym.make('CartPole-v1')
 
# 4. Initialize the predictive model and self-supervised RL agent
model = PredictiveModel(input_size=env.observation_space.shape[0], output_size=env.observation_space.shape[0])
agent = SelfSupervisedRLAgent(model)
 
# 5. Collect data from the environment to simulate self-supervised learning
num_episodes = 100
for episode in range(num_episodes):
    state = env.reset()
    state = torch.tensor(state, dtype=torch.float32)
    done = False
    total_reward = 0
    states = []
    next_states = []
 
    while not done:
        # Simulate agent interaction with the environment
        action = env.action_space.sample()  # Random action for exploration
        next_state, reward, done, _, _ = env.step(action)
 
        states.append(state)  # Store current state
        next_states.append(torch.tensor(next_state, dtype=torch.float32))  # Store next state
 
        state = torch.tensor(next_state, dtype=torch.float32)
        total_reward += reward
 
    # 6. Train the agent using self-supervised learning
    states = torch.stack(states)
    next_states = torch.stack(next_states)
    loss = agent.train(states, next_states)
 
    if episode % 10 == 0:
        print(f"Episode {episode}, Loss: {loss:.4f}, Total Reward: {total_reward}")
 
# 7. Evaluate the model's prediction
state = env.reset()
state = torch.tensor(state, dtype=torch.float32)
predicted_next_state = agent.predict_next_state(state)
 
print(f"Predicted next state: {predicted_next_state}")