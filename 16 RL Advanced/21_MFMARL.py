"""
Project 621: Mean Field Multi-agent Reinforcement Learning
Description:
Mean field multi-agent reinforcement learning (MF-MARL) is an approach where each agent in a multi-agent system approximates the effect of all other agents using a mean field—a simplified aggregate representation. This method reduces the complexity of large-scale multi-agent systems by allowing agents to consider the average behavior of others instead of explicitly interacting with every agent. In this project, we will implement mean field MARL, where each agent makes decisions based on the average impact of other agents in the environment.
"""

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
 
# 1. Define the neural network model for mean field multi-agent reinforcement learning
class MeanFieldPolicyNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(MeanFieldPolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_size)  # Output: action probabilities
 
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return torch.softmax(self.fc3(x), dim=-1)  # Softmax for action probabilities
 
# 2. Define the Mean Field MARL agent
class MeanFieldMARLAgent:
    def __init__(self, model, learning_rate=0.001, gamma=0.99, epsilon=1.0, epsilon_decay=0.995):
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.criterion = nn.MSELoss()
 
    def select_action(self, state):
        # Select an action based on the policy's probability distribution
        action_probs = self.model(torch.tensor(state, dtype=torch.float32))
        action = np.random.choice(len(action_probs), p=action_probs.detach().numpy())
        return action
 
    def update(self, state, action, reward, next_state, done, mean_field):
        # Mean-field Q-learning update rule: Adjust reward using mean-field approximation
        mean_field_reward = reward + mean_field  # Adjust reward using the average behavior of other agents
 
        # Q-learning update rule
        q_values = self.model(torch.tensor(state, dtype=torch.float32))
        next_q_values = self.model(torch.tensor(next_state, dtype=torch.float32))
        target = mean_field_reward + self.gamma * torch.max(next_q_values) * (1 - done)
        loss = self.criterion(q_values[action], target)  # Compute loss (MSE)
 
        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
 
        # Decay epsilon (exploration rate)
        if done:
            self.epsilon *= self.epsilon_decay
 
        return loss.item()
 
# 3. Initialize the environment and Mean Field MARL agent
env = gym.make('CartPole-v1')
model = MeanFieldPolicyNetwork(input_size=env.observation_space.shape[0], output_size=env.action_space.n)
agent = MeanFieldMARLAgent(model)
 
# 4. Train the agent using Mean Field Multi-agent RL
num_episodes = 1000
for episode in range(num_episodes):
    state = env.reset()
    done = False
    total_reward = 0
    mean_field = 0  # Mean field representation (average behavior of other agents)
    states = []
    actions = []
    rewards = []
 
    while not done:
        action = agent.select_action(state)
        next_state, reward, done, _, _ = env.step(action)
 
        # Update mean field as the average reward from other agents
        mean_field = np.mean(rewards)  # Here, we use rewards as a simple proxy for other agents' actions
 
        # Collect states, actions, and rewards
        states.append(state)
        actions.append(action)
        rewards.append(reward)
 
        state = next_state
        total_reward += reward
 
    # Update the model using the mean-field approximation of other agents
    loss = agent.update(states, actions, rewards, next_state, done, mean_field)
 
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
 
print(f"Total reward after Mean Field MARL training: {total_reward}")
