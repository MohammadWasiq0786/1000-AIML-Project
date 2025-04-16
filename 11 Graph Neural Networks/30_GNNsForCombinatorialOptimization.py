"""
Project 430. GNNs for combinatorial optimization
Description:
Combinatorial optimization problems like the Traveling Salesman Problem (TSP) or Max-Cut can be modeled using graphs. Traditional solvers are slow for large instances, but GNNs can learn to approximate solutions using patterns in the graph structure. In this project, we’ll implement a Graph Pointer Network to find approximate TSP tours using a learned GNN policy.

About:
✅ What It Does:
Simulates a random TSP instance as a graph.

Uses a GCN to embed node positions.

Applies a greedy decoding strategy to generate a tour.

Minimizes tour length and visualizes the result.
"""

import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import numpy as np
import matplotlib.pyplot as plt
 
# 1. Generate a random TSP graph
def generate_tsp_graph(n_cities=10):
    coords = torch.rand(n_cities, 2)
    edge_index = torch.combinations(torch.arange(n_cities), r=2).t()
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)  # undirected
    return coords, edge_index
 
# 2. GCN-based TSP embedding model
class TSPGNN(torch.nn.Module):
    def __init__(self, in_dim=2, hidden_dim=64):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.out = Linear(hidden_dim, 1)
 
    def forward(self, x, edge_index):
        h = F.relu(self.conv1(x, edge_index))
        h = F.relu(self.conv2(h, edge_index))
        return h
 
# 3. Greedy tour decoder using node embeddings
def decode_tour(embeddings):
    n = embeddings.size(0)
    visited = [0]
    for _ in range(n - 1):
        last = visited[-1]
        dists = torch.norm(embeddings[last] - embeddings, dim=1)
        dists[visited] = float('inf')  # mask visited
        next_node = dists.argmin().item()
        visited.append(next_node)
    return visited
 
# 4. Train the model to minimize tour length (supervised or RL can be added)
def tsp_loss(coords, tour):
    return sum(torch.norm(coords[tour[i]] - coords[tour[i + 1]]) for i in range(len(tour) - 1)) + torch.norm(coords[tour[-1]] - coords[tour[0]])
 
# 5. Main execution
coords, edge_index = generate_tsp_graph()
data = Data(x=coords, edge_index=edge_index)
model = TSPGNN()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
 
# Training loop (toy example, not a true TSP solver yet)
for epoch in range(1, 101):
    model.train()
    optimizer.zero_grad()
    emb = model(data.x, data.edge_index)
    tour = decode_tour(emb.detach())
    loss = tsp_loss(data.x, tour)
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0:
        print(f"Epoch {epoch:03d}, Tour Length: {loss.item():.4f}")
 
# 6. Visualize tour
plt.scatter(data.x[:, 0], data.x[:, 1], color='red')
for i in range(len(tour)):
    a, b = data.x[tour[i]], data.x[tour[(i+1)%len(tour)]]
    plt.plot([a[0], b[0]], [a[1], b[1]], 'b-')
plt.title("TSP Tour Found by GNN")
plt.show()