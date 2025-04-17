"""
Project 623: Relational Reinforcement Learning
Description:
Relational reinforcement learning (Relational RL) extends traditional RL by incorporating relations between entities in the environment. In environments with multiple entities (e.g., agents, objects, or nodes), relationships between these entities can significantly impact decision-making. Relational RL uses relational models to understand the interactions between these entities and learns policies that consider these relationships. In this project, we will implement relational reinforcement learning, where the agent learns to make decisions based on the relational structure of the environment.
"""

import gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import networkx as nx
 
# 1. Define the neural network model for relational reinforcement learning
class RelationalPolicyNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(RelationalPolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_size)  # Output: action probabilities
 
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return torch.softmax(self.fc3(x), dim=-1)  # Softmax for action probabilities
 
# 2. Define the Relational Reinforcement Learning Agent
class RelationalRLAgent:
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
 
    def update(self, state, action, reward, next_state, done, relational_features):
        # Incorporate relational features (e.g., interaction between agents or objects)
        reward += np.dot(relational_features, np.array([0.5, -0.5]))  # Example relation shaping
 
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
 
# 3. Define a simple relational environment (graph-based)
class RelationalGraphEnv:
    def __init__(self):
        self.graph = nx.erdos_renyi_graph(10, 0.3)  # Random graph with 10 nodes and edge probability of 0.3
        self.state = np.array([np.random.rand(3) for _ in range(len(self.graph.nodes))])  # Random node features
        self.num_nodes = len(self.graph.nodes)
    
    def reset(self):
        # Reset the environment and return the initial state
        self.graph = nx.erdos_renyi_graph(10, 0.3)
        self.state = np.array([np.random.rand(3) for _ in range(len(self.graph.nodes))])
        return self.state
 
    def step(self, action):
        # Simulate the environment step based on the action taken
        if action == 0:
            node1, node2 = np.random.choice(self.num_nodes, size=2, replace=False)
            self.graph.add_edge(node1, node2)
        elif action == 1:
            edges = list(self.graph.edges)
            if edges:
                edge = np.random.choice(len(edges))
                self.graph.remove_edge(*edges[edge])
        
        # Calculate reward (e.g., based on the number of edges in the graph)
        reward = len(self.graph.edges)
        
        done = False  # Environment doesn't end in this simple case
        
        # Example of relational features: average node degree
        relational_features = np.mean([deg for _, deg in self.graph.degree()])  # Average degree of nodes
        
        return self.state, reward, done, relational_features
 
# 4. Initialize the environment and relational RL agent
env = RelationalGraphEnv()
model = RelationalPolicyNetwork(input_size=env.state.shape[1], output_size=2)  # 2 actions (add/remove edge)
agent = RelationalRLAgent(model)
 
# 5. Train the agent using Relational Reinforcement Learning
num_episodes = 1000
for episode in range(num_episodes):
    state = env.reset()
    done = False
    total_reward = 0
    relational_features = []
 
    while not done:
        action = agent.select_action(state)
        next_state, reward, done, relation = env.step(action)
 
        # Collect relational features
        relational_features.append(relation)
 
        # Update the agent using relational RL
        loss = agent.update(state, action, reward, next_state, done, relational_features)
 
        state = next_state
        total_reward += reward
 
    if episode % 100 == 0:
        print(f"Episode {episode}, Total Reward: {total_reward}, Loss: {loss:.4f}")
 
# 6. Evaluate the agent after training (no exploration, only exploitation)
state = env.reset()
done = False
total_reward = 0
while not done:
    action = agent.select_action(state)
    next_state, reward, done, _ = env.step(action)
    total_reward += reward
    state = next_state
 
print(f"Total reward after Relational RL training: {total_reward}")