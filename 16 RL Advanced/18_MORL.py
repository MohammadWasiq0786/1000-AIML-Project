"""
Project 618: Multi-objective Reinforcement Learning
Description:
Multi-objective reinforcement learning (MORL) deals with problems where the agent needs to optimize multiple objectives simultaneously, which may conflict with each other. The challenge is to find a trade-off between these objectives. In this project, we will implement MORL where the agent optimizes two or more objectives, balancing the rewards for each objective.
"""

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
 
# 1. Define the policy network for multi-objective RL
class MultiObjectivePolicyNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(MultiObjectivePolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_size)  # Output: action probabilities
 
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return torch.softmax(self.fc3(x), dim=-1)  # Softmax for action probabilities
 
# 2. Define the agent for multi-objective reinforcement learning (using weighted sum of objectives)
class MultiObjectiveRLAgent:
    def __init__(self, model, learning_rate=0.001, objectives_weights=[0.5, 0.5]):
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.objectives_weights = objectives_weights  # Weights for each objective
 
    def select_action(self, state):
        # Select an action based on the policy's probability distribution
        action_probs = self.model(torch.tensor(state, dtype=torch.float32))
        action = np.random.choice(len(action_probs), p=action_probs.detach().numpy())
        return action
 
    def update(self, states, actions, rewards, done):
        # Update the model using the weighted sum of rewards from multiple objectives
        self.optimizer.zero_grad()
 
        # Compute the weighted sum of rewards from each objective
        weighted_reward = sum(w * r for w, r in zip(self.objectives_weights, rewards))
 
        # Compute loss (negative log-likelihood of the selected action)
        action_probs = self.model(states)
        log_probs = torch.log(action_probs.gather(1, actions.unsqueeze(1)))
        loss = -torch.mean(log_probs * weighted_reward)  # Maximize weighted reward
 
        # Backpropagation
        loss.backward()
        self.optimizer.step()
 
        return loss.item()
 
# 3. Initialize the environment and agent
env = gym.make('CartPole-v1')
model = MultiObjectivePolicyNetwork(input_size=env.observation_space.shape[0], output_size=env.action_space.n)
agent = MultiObjectiveRLAgent(model)
 
# 4. Train the agent using multi-objective reinforcement learning (weighted sum approach)
num_episodes = 1000
for episode in range(num_episodes):
    state = env.reset()
    done = False
    total_reward = [0, 0]  # Two objectives
    states = []
    actions = []
    rewards = []
 
    while not done:
        action = agent.select_action(state)
        next_state, reward, done, _, _ = env.step(action)
 
        # Define two different reward objectives
        # Objective 1: Total reward
        # Objective 2: Stay within a certain position range
        reward_objective_1 = reward
        reward_objective_2 = 1 if abs(state[0]) < 0.1 else 0  # Reward for staying within position range
 
        # Collect states, actions, and rewards
        states.append(state)
        actions.append(action)
        rewards.append([reward_objective_1, reward_objective_2])
 
        state = next_state
        total_reward[0] += reward_objective_1
        total_reward[1] += reward_objective_2
 
    # Update the model using the weighted sum of rewards
    loss = agent.update(torch.tensor(states, dtype=torch.float32), torch.tensor(actions), rewards, done)
 
    if episode % 100 == 0:
        print(f"Episode {episode}, Total Reward (Objective 1, Objective 2): {total_reward}, Loss: {loss:.4f}")
 
# 5. Evaluate the agent after training (no exploration, only exploitation)
state = env.reset()
done = False
total_reward = [0, 0]
while not done:
    action = agent.select_action(state)
    next_state, reward, done, _, _ = env.step(action)
    total_reward[0] += reward
    total_reward[1] += 1 if abs(state[0]) < 0.1 else 0
    state = next_state
 
print(f"Total reward after Multi-objective RL training: {total_reward}")