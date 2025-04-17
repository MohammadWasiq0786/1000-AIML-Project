"""
Project 619: Cooperative Multi-agent Reinforcement Learning Using DQN
Description:
Cooperative multi-agent reinforcement learning (CMARL) focuses on scenarios where multiple agents work together to achieve a common goal. The agents share information and learn from each other’s experiences to maximize a joint reward function. In this project, we will implement cooperative multi-agent reinforcement learning, where agents interact in a shared environment, learning to cooperate for a collective objective.
"""

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
 
# 1. Define the neural network model for the agent (DQN for cooperative agents)
class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_size)  # Output: action probabilities
 
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)  # Linear output for Q-values
 
# 2. Define the Cooperative Multi-agent Reinforcement Learning Agent
class CooperativeMARLAgent:
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
 
# 3. Initialize the environment (Cooperative setting with multiple agents)
env = gym.make('CartPole-v1')
agent1 = CooperativeMARLAgent(DQN(input_size=env.observation_space.shape[0], output_size=env.action_space.n))
agent2 = CooperativeMARLAgent(DQN(input_size=env.observation_space.shape[0], output_size=env.action_space.n))
 
# 4. Train the agents in a cooperative environment
num_episodes = 1000
for episode in range(num_episodes):
    state = env.reset()
    done = False
    total_reward = 0
    while not done:
        # Each agent selects its action
        action1 = agent1.select_action(state)
        action2 = agent2.select_action(state)
 
        # Perform the joint actions (agents act simultaneously)
        next_state, reward, done, _, _ = env.step([action1, action2])
 
        # Each agent updates its Q-values based on the shared reward
        total_reward += reward
        loss1 = agent1.update(state, action1, reward, next_state, done)
        loss2 = agent2.update(state, action2, reward, next_state, done)
 
        state = next_state
 
    # Print training progress every 100 episodes
    if episode % 100 == 0:
        print(f"Episode {episode}, Total Reward: {total_reward}, Loss1: {loss1:.4f}, Loss2: {loss2:.4f}")
 
# 5. Evaluate the cooperative agents after training
state = env.reset()
done = False
total_reward = 0
while not done:
    action1 = agent1.select_action(state)
    action2 = agent2.select_action(state)
    next_state, reward, done, _, _ = env.step([action1, action2])
    total_reward += reward
    state = next_state
 
print(f"Total reward after Cooperative RL training: {total_reward}")