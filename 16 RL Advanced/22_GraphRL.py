"""
Project 622: Graph Reinforcement Learning
Description:
Graph reinforcement learning (Graph RL) involves applying reinforcement learning to environments represented as graphs. In graph-based environments, the state space can be represented as nodes and edges, and agents take actions that influence the structure or properties of the graph. This project will explore graph RL, where an agent learns to make decisions that affect the graph structure, such as node classification, link prediction, or graph traversal.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import networkx as nx
 
# 1. Define a simple Graph Neural Network (GNN) model for Graph RL
class GraphNeuralNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(GraphNeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_size)  # Output: action probabilities
 
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)  # Linear output for Q-values
 
# 2. Define the Graph RL agent using the GNN model
class GraphRLAgent:
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
 
# 3. Define a simple graph environment (graph-based environment for RL)
class GraphEnv:
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
        # Here we simulate an environment step based on the action taken
        # Action: 0 = add an edge, 1 = remove an edge
        if action == 0:
            # Add an edge (for simplicity, randomly pick two nodes)
            node1, node2 = np.random.choice(self.num_nodes, size=2, replace=False)
            self.graph.add_edge(node1, node2)
        elif action == 1:
            # Remove an edge (for simplicity, randomly pick an edge)
            edges = list(self.graph.edges)
            if edges:
                edge = np.random.choice(len(edges))
                self.graph.remove_edge(*edges[edge])
        
        # Calculate reward (for example, based on the number of edges in the graph)
        reward = len(self.graph.edges)  # Higher reward for more edges
        
        done = False  # Assume the environment doesn't end after each step
        
        return self.state, reward, done
 
# 4. Initialize the graph environment and RL agent
env = GraphEnv()
model = GraphNeuralNetwork(input_size=3, output_size=2)  # 3 features per node, 2 possible actions (add or remove edge)
agent = GraphRLAgent(model)
 
# 5. Train the agent using Graph RL
num_episodes = 1000
for episode in range(num_episodes):
    state = env.reset()
    done = False
    total_reward = 0
    while not done:
        action = agent.select_action(state)
        next_state, reward, done = env.step(action)
        total_reward += reward
        
        # Update the agent using the collected experience
        loss = agent.update(state, action, reward, next_state, done)
        state = next_state
 
    # Print training progress every 100 episodes
    if episode % 100 == 0:
        print(f"Episode {episode}, Total Reward: {total_reward}, Loss: {loss:.4f}")
 
# 6. Evaluate the agent after training (no exploration, only exploitation)
state = env.reset()
done = False
total_reward = 0
while not done:
    action = agent.select_action(state)
    next_state, reward, done = env.step(action)
    total_reward += reward
    state = next_state
 
print(f"Total reward after Graph RL training: {total_reward}")