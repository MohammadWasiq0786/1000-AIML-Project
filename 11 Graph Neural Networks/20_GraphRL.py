"""
Project 420. Graph reinforcement learning
Description:
Graph Reinforcement Learning (Graph RL) integrates reinforcement learning into graph-based environments. It’s used in molecule generation, robot navigation on graphs, and optimization problems (like traveling salesman). The agent learns policies over a graph structure — for example, traversing a network or building a solution step-by-step.

In this project, we’ll implement a simple Q-learning agent that learns to navigate a graph maze to reach a goal.

About:
✅ What It Does:
Models a graph maze using NetworkX.

Trains a Q-learning agent to navigate from any node to a goal node.

Uses Q-table to store action values for each graph edge.

Visualizes the learned shortest path after training.
"""


# pip install networkx matplotlib

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import random
 
# 1. Create a simple graph as environment
G = nx.grid_2d_graph(5, 5)  # 5x5 grid
G = nx.convert_node_labels_to_integers(G)
state_space = list(G.nodes())
action_space = {s: list(G.neighbors(s)) for s in state_space}
goal_state = 24  # Bottom-right corner
 
# 2. Initialize Q-table
Q = np.zeros((len(state_space), len(state_space)))
alpha = 0.1     # learning rate
gamma = 0.9     # discount factor
epsilon = 0.2   # exploration rate
 
# 3. Define reward function
def reward(s):
    return 10 if s == goal_state else -1
 
# 4. Q-learning loop
for episode in range(1000):
    state = random.choice(state_space)
    while state != goal_state:
        # Epsilon-greedy action
        if random.random() < epsilon:
            next_state = random.choice(action_space[state])
        else:
            next_state = max(action_space[state], key=lambda a: Q[state, a])
 
        r = reward(next_state)
        Q[state, next_state] += alpha * (r + gamma * np.max(Q[next_state]) - Q[state, next_state])
        state = next_state
 
# 5. Test learned policy from node 0
path = [0]
current = 0
while current != goal_state:
    next_node = max(action_space[current], key=lambda a: Q[current, a])
    path.append(next_node)
    current = next_node
 
print("Learned path from node 0 to goal:", path)
 
# 6. Visualize graph and path
pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True, node_color='lightgray')
nx.draw_networkx_nodes(G, pos, nodelist=path, node_color='orange')
nx.draw_networkx_edges(G, pos, edgelist=[(path[i], path[i+1]) for i in range(len(path)-1)], edge_color='red', width=2)
plt.title("Graph RL: Learned Path from Node 0 to Goal")
plt.show()