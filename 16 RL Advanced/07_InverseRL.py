"""
Project 607: Inverse Reinforcement Learning
Description:
Inverse Reinforcement Learning (IRL) involves inferring the reward function from observed behavior. Instead of learning the policy directly from the environment, an agent learns what reward function would make the observed behavior optimal. This is useful in applications where the reward function is hard to define but expert demonstrations are available. In this project, we will implement IRL using a simple framework to infer the reward function from expert trajectories.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gym
 
# 1. Define a simple MLP to represent the reward function
class RewardModel(nn.Module):
    def __init__(self, input_size):
        super(RewardModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 1)  # Output a single scalar (reward)
 
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)
 
# 2. Define the IRL agent
class IRLAgent:
    def __init__(self, reward_model, learning_rate=0.001):
        self.reward_model = reward_model
        self.optimizer = optim.Adam(reward_model.parameters(), lr=learning_rate)
 
    def train(self, trajectories, expert_rewards):
        self.reward_model.train()
        total_loss = 0
 
        # Process the expert trajectories and rewards
        for trajectory, reward in zip(trajectories, expert_rewards):
            states, actions = zip(*trajectory)
            states = torch.tensor(states, dtype=torch.float32)
 
            # Get reward prediction from the model
            predicted_rewards = self.reward_model(states)
            loss = ((predicted_rewards - reward) ** 2).mean()  # MSE loss
            total_loss += loss.item()
 
            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
 
        return total_loss
 
# 3. Load the environment and expert demonstrations
env = gym.make('CartPole-v1')
 
# Expert agent demonstrations (random behavior for simplicity)
expert_trajectories = []
expert_rewards = []
 
for episode in range(10):  # Collect data from 10 episodes
    state = env.reset()
    done = False
    trajectory = []
    total_reward = 0
    while not done:
        action = env.action_space.sample()  # Expert's random action
        next_state, reward, done, _, _ = env.step(action)
        trajectory.append((state, action))
        total_reward += reward
        state = next_state
    expert_trajectories.append(trajectory)
    expert_rewards.append(total_reward)
 
# 4. Initialize the reward model and IRL agent
reward_model = RewardModel(input_size=env.observation_space.shape[0])
irl_agent = IRLAgent(reward_model)
 
# 5. Train the IRL agent
num_epochs = 100
for epoch in range(num_epochs):
    loss = irl_agent.train(expert_trajectories, expert_rewards)
    print(f"Epoch {epoch+1}, Loss: {loss}")
 
# 6. Evaluate the learned reward function
state = env.reset()
done = False
total_reward = 0
while not done:
    state_tensor = torch.tensor(state, dtype=torch.float32)
    reward = reward_model(state_tensor).item()  # Get predicted reward
    action = env.action_space.sample()  # Take random action (for simplicity)
    next_state, _, done, _, _ = env.step(action)
    total_reward += reward
    state = next_state
 
print(f"Total reward after IRL: {total_reward}")