"""
Project 438. Graph contrastive learning
Description:
Graph Contrastive Learning (GCL) is a self-supervised learning method that trains GNNs by comparing different views of the same graph — maximizing agreement between positive pairs (e.g., two augmentations of the same node or graph) and minimizing it for negative pairs. This leads to strong representations even without labels. In this project, we implement a simple GraphCL-style node-level contrastive learning using random augmentations.

About:
✅ What It Does:
Applies node dropout to generate two augmented graph views.

Trains a GCN encoder to produce similar embeddings for the same nodes in both views.

Uses contrastive loss to align positive pairs and separate negative ones.

Learns unsupervised node representations suitable for downstream tasks (e.g., node classification).
"""

import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv
from torch_geometric.utils import dropout_node
from torch.nn import CosineSimilarity
 
# 1. Load dataset
dataset = Planetoid(root="/tmp/Cora", name="Cora")
data = dataset[0]
 
# 2. Define GCN encoder
class GCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
 
    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        return self.conv2(x, edge_index)
 
# 3. Model setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GCNEncoder(dataset.num_node_features, 64).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
similarity = CosineSimilarity(dim=1)
 
# 4. Contrastive loss: maximize agreement of node embeddings from two augmentations
def contrastive_loss(z1, z2, temperature=0.5):
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    sim_matrix = torch.matmul(z1, z2.T) / temperature
    labels = torch.arange(z1.size(0)).to(device)
    return F.cross_entropy(sim_matrix, labels)
 
# 5. Training loop
data = data.to(device)
for epoch in range(1, 101):
    model.train()
    optimizer.zero_grad()
 
    # Two augmented views via random node dropout
    view1, _ = dropout_node(data, p=0.2)
    view2, _ = dropout_node(data, p=0.2)
 
    z1 = model(view1.x, view1.edge_index)
    z2 = model(view2.x, view2.edge_index)
 
    loss = contrastive_loss(z1, z2)
    loss.backward()
    optimizer.step()
 
    if epoch % 10 == 0:
        print(f"Epoch {epoch:03d}, Contrastive Loss: {loss.item():.4f}")