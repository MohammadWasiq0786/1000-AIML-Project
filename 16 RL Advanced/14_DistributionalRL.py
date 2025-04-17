"""
Project 614: Distributional Reinforcement Learning
Description:
Distributional reinforcement learning (DRL) aims to model the distribution of returns (rewards) rather than just the expected return. By learning the full distribution of returns, the agent can make better decisions, especially in environments with high variability. In this project, we will implement distributional Q-learning, where the agent learns the entire distribution of returns for each action, rather than just the expected value.
"""

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
 
# 1. Define the distributional Q-learning model (C51 variant)
class C51Model(nn.Module):
    def __init__(self, input_size, output_size, n_atoms=51, v_min=-10, v_max=10):
        super(C51Model, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_size * n_atoms)  # For each action, we have n_atoms probabilities
 
        self.n_atoms = n_atoms
        self.v_min = v_min
        self.v_max = v_max
        self.support = torch.linspace(self.v_min, self.v_max, self.n_atoms)  # Discretized support for the value distribution
 
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x).view(-1, len(x), self.n_atoms)  # Reshape to (batch_size, actions, atoms)
        return x
 
    def get_q_values(self, x):
        probs = self.forward(x)
        q_values = torch.sum(probs * self.support, dim=2)  # Compute expected value from distribution
        return q_values
 
# 2. Define the agent for distributional Q-learning
class C51Agent:
    def __init__(self, model, learning_rate=0.001, gamma=0.99, n_atoms=51):
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.gamma = gamma
        self.n_atoms = n_atoms
        self.criterion = nn.KLDivLoss(reduction='batchmean')  # Kullback-Leibler divergence for distribution matching
 
    def update(self, states, actions, rewards, next_states, dones):
        self.model.train()
        batch_size = len(states)
        next_probs = self.model(next_states)
 
        # Compute target distribution (Bellman backup for distribution)
        next_q_values = self.model.get_q_values(next_states)
        next_action_values = next_q_values.max(1)[0].unsqueeze(1)  # Max Q-value for each next state
        next_action_probabilities = (next_probs * next_action_values).sum(dim=2)
 
        target = rewards + (self.gamma * next_action_probabilities * (1 - dones))
        target_distribution = target.unsqueeze(1).expand_as(next_probs)  # Expand target distribution for each action
 
        # Compute loss (KL divergence between predicted and target distributions)
        loss = self.criterion(torch.log(next_probs), target_distribution)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
 
# 3. Initialize the environment and agent
env = gym.make('CartPole-v1')
model = C51Model(input_size=env.observation_space.shape[0], output_size=env.action_space.n)
agent = C51Agent(model)
 
# 4. Train the agent with distributional Q-learning (C51)
num_episodes = 1000
for episode in range(num_episodes):
    state = env.reset()
    done = False
    total_reward = 0
    states = []
    actions = []
    rewards = []
    next_states = []
    dones = []
 
    while not done:
        action_probs = model(torch.tensor(state, dtype=torch.float32).unsqueeze(0))
        action = np.random.choice(env.action_space.n, p=action_probs.detach().numpy()[0])
        next_state, reward, done, _ = env.step(action)
 
        # Collect experience
        states.append(state)
        actions.append(action)
        rewards.append(reward)
        next_states.append(next_state)
        dones.append(done)
 
        state = next_state
        total_reward += reward
 
    # Update the model using the collected batch of experiences
    loss = agent.update(states, actions, rewards, next_states, dones)
 
    if episode % 100 == 0:
        print(f"Episode {episode}, Total Reward: {total_reward}, Loss: {loss}")
 
# 5. Evaluate the agent after training (no exploration, only exploitation)
state = env.reset()
done = False
total_reward = 0
while not done:
    action_probs = model(torch.tensor(state, dtype=torch.float32).unsqueeze(0))
    action = np.argmax(action_probs.detach().numpy())
    next_state, reward, done, _ = env.step(action)
    total_reward += reward
    state = next_state
 
print(f"Total reward after Distributional RL training: {total_reward}")