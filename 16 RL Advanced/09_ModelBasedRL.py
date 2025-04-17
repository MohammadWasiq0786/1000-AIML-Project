"""
Project 609: Model-based Reinforcement Learning
Description:
Model-based reinforcement learning (MBRL) involves learning a model of the environment's dynamics (i.e., how the state evolves when actions are taken). Instead of relying solely on trial and error, the agent can plan ahead by using this learned model to predict future states and rewards. In this project, we will implement MBRL by learning a dynamics model of the environment and using it to plan actions.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gym
 
# 1. Define a simple neural network model to predict the next state
class DynamicsModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(DynamicsModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_size)  # Output: Predicted next state
 
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)
 
# 2. Define the model-based reinforcement learning agent
class ModelBasedRLAgent:
    def __init__(self, dynamics_model, learning_rate=0.001):
        self.dynamics_model = dynamics_model
        self.optimizer = optim.Adam(dynamics_model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
 
    def train(self, states, next_states):
        self.dynamics_model.train()
        self.optimizer.zero_grad()
 
        # Predict the next state using the learned dynamics model
        predicted_next_state = self.dynamics_model(states)
 
        # Calculate the loss between predicted next state and actual next state
        loss = self.criterion(predicted_next_state, next_states)
        loss.backward()
        self.optimizer.step()
        return loss.item()
 
    def predict_next_state(self, state):
        self.dynamics_model.eval()
        with torch.no_grad():
            return self.dynamics_model(state)
 
# 3. Load the environment
env = gym.make('CartPole-v1')
 
# 4. Initialize the dynamics model and model-based RL agent
model = DynamicsModel(input_size=env.observation_space.shape[0], output_size=env.observation_space.shape[0])
agent = ModelBasedRLAgent(model)
 
# 5. Collect data from the environment to simulate model-based learning
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
 
    # 6. Train the agent using the learned dynamics model
    states = torch.stack(states)
    next_states = torch.stack(next_states)
    loss = agent.train(states, next_states)
 
    if episode % 10 == 0:
        print(f"Episode {episode}, Loss: {loss:.4f}, Total Reward: {total_reward}")
 
# 7. Evaluate the model's ability to predict the next state
state = env.reset()
state = torch.tensor(state, dtype=torch.float32)
predicted_next_state = agent.predict_next_state(state)
 
print(f"Predicted next state: {predicted_next_state}")