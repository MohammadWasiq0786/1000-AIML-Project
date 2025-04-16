"""
Project 440. Graph attention visualization
Description:
Graph Attention Networks (GATs) assign learnable weights to edges — letting the model focus on more relevant neighbors. Visualizing these attention weights gives insight into which nodes influence predictions, helping with interpretability in tasks like fraud detection, molecule analysis, or social network reasoning. In this project, we’ll train a GAT on the Cora dataset and visualize the learned node-level attention.

About:
✅ What It Does:
Trains a Graph Attention Network on the Cora dataset.

Extracts attention weights from node 10 to its neighbors.

Builds and visualizes a subgraph, with edge colors representing attention strength.

Provides visual insight into how the model makes decisions.
"""

# pip install torch-geometric matplotlib networkx

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GATConv
 
# 1. Load Cora dataset
dataset = Planetoid(root='/tmp/Cora', name='Cora')
data = dataset[0]
 
# 2. Define GAT model (single-head for easy visualization)
class GAT(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.gat = GATConv(in_channels, out_channels, heads=1, concat=False)
 
    def forward(self, x, edge_index):
        return self.gat(x, edge_index)
 
# 3. Setup model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GAT(dataset.num_node_features, dataset.num_classes).to(device)
data = data.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
loss_fn = torch.nn.CrossEntropyLoss()
 
# 4. Training loop
def train():
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = loss_fn(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
 
for epoch in range(1, 101):
    train()
 
# 5. Visualize attention around a specific node
node_idx = 10
model.eval()
_, attn_weights = model.gat(data.x, data.edge_index, return_attention_weights=True)
edge_index, alpha = attn_weights
alpha = alpha.squeeze()
 
# Get subgraph centered on node_idx
neighbors = edge_index[1][edge_index[0] == node_idx]
edges = [(node_idx, int(n)) for n in neighbors.cpu()]
weights = [float(alpha[i]) for i in range(len(alpha)) if edge_index[0][i] == node_idx]
 
# Draw attention graph
G = nx.Graph()
G.add_node(node_idx)
for (n, w) in zip(neighbors.tolist(), weights):
    G.add_edge(node_idx, n, weight=w)
 
pos = nx.spring_layout(G)
edges, weights = zip(*nx.get_edge_attributes(G, 'weight').items())
nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color=weights, width=4.0, edge_cmap=plt.cm.Blues)
plt.title(f"GAT Attention from Node {node_idx}")
plt.show()