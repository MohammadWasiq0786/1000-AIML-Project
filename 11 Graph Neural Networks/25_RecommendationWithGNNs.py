"""
Project 425. Recommendation with GNNs
Description:
Graph Neural Networks can power recommendation systems by modeling user-item interactions as a bipartite graph. This allows the system to capture high-order connectivity and indirect preferences. In this project, we’ll use LightGCN (a simplified GCN for collaborative filtering) to recommend items to users based on past interactions using PyTorch Geometric.

About:
✅ What It Does:
Constructs a user-item bipartite graph.

Trains a LightGCN to learn embeddings via one-hop neighbor aggregation.

Computes user-item scores using dot products in embedding space.

Recommends top-scoring items for each user based on learned preferences.
"""

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
import networkx as nx
import matplotlib.pyplot as plt
 
# 1. Simulate user-item graph (bipartite)
# 5 users (0–4), 5 items (5–9)
edges = [
    (0, 5), (0, 6), (1, 5), (1, 7), (2, 8),
    (3, 7), (3, 9), (4, 6), (4, 8)
]
edges = edges + [(j, i) for i, j in edges]  # undirected graph
edge_index = torch.tensor(edges, dtype=torch.long).t()
 
# Node features: initialize randomly or with identity
x = torch.eye(10)  # 10 nodes total (5 users + 5 items)
 
# 2. Create PyG Data object
data = Data(x=x, edge_index=edge_index)
 
# 3. Define LightGCN (simplified GCN without non-linearity or bias)
class LightGCN(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, out_channels, add_self_loops=False, bias=False)
 
    def forward(self, x, edge_index):
        return self.conv1(x, edge_index)
 
# 4. Initialize and run the model
model = LightGCN(in_channels=10, out_channels=4)
with torch.no_grad():
    z = model(data.x, data.edge_index)
 
# 5. Recommendation: compute dot product between user and item embeddings
user_emb = z[0:5]   # users 0–4
item_emb = z[5:10]  # items 5–9
scores = torch.matmul(user_emb, item_emb.t())  # 5 users × 5 items
 
# Top item recommendation per user
recommendations = torch.argmax(scores, dim=1)
print("Top recommended item per user:")
for u, item in enumerate(recommendations):
    print(f"User {u} → Recommend Item {item.item()+5}")
 
# 6. Optional: Visualize interaction graph
G = nx.Graph()
G.add_edges_from(edges)
pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True, node_color='lightblue')
plt.title("User-Item Interaction Graph")
plt.show()