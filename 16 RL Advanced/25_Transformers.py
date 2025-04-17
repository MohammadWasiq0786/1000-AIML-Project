"""
Project 625: Transformers for Reinforcement Learning
Description:
Transformers have become the architecture of choice for many sequence-based tasks in deep learning, and they can also be applied to reinforcement learning (RL). By using transformers, RL agents can model long-range dependencies in their states, actions, and rewards, allowing them to handle more complex decision-making problems. In this project, we will explore the use of transformers in RL, focusing on using transformer-based models to improve decision-making processes by capturing sequential patterns and long-term dependencies.
"""

import gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
 
# 1. Define the Transformer-based model for RL
class TransformerPolicyNetwork(nn.Module):
    def __init__(self, input_size, output_size, num_heads=4, num_layers=2):
        super(TransformerPolicyNetwork, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        
        # Transformer encoder layer
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=input_size, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, output_size)  # Output: action probabilities
 
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        
        # Reshape input to (sequence_length, batch_size, feature_size) for transformer
        x = x.unsqueeze(0)  # Adding batch dimension (1 batch here)
        x = self.transformer_encoder(x)  # Apply transformer encoder
        
        x = x.squeeze(0)  # Remove batch dimension
        return torch.softmax(self.fc2(x), dim=-1)  # Output action probabilities
 
# 2. Define the RL agent using the Transformer-based model
class TransformerRLAgent:
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
 
# 3. Initialize the environment and Transformer RL agent
env = gym.make('CartPole-v1')
model = TransformerPolicyNetwork(input_size=env.observation_space.shape[0], output_size=env.action_space.n)
agent = TransformerRLAgent(model)
 
# 4. Train the agent using Transformer-based RL
num_episodes = 1000
for episode in range(num_episodes):
    state = env.reset()
    done = False
    total_reward = 0
    while not done:
        action = agent.select_action(state)
        next_state, reward, done, _, _ = env.step(action)
 
        # Update the agent using the transformer-based policy network
        loss = agent.update(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward
 
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
 
print(f"Total reward after Transformer RL training: {total_reward}")