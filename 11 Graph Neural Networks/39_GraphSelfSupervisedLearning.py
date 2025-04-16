"""
Project 439. Graph self-supervised learning
Description:
Graph self-supervised learning (SSL) enables models to learn meaningful node or graph representations without labels by solving pretext tasks like node masking, context prediction, or contrastive learning. These learned embeddings can be transferred to downstream tasks (e.g., classification or clustering). In this project, we’ll implement a node attribute masking SSL task using a GCN encoder.

About:
✅ What It Does:
Randomly masks a subset of node features.

Trains a GCN encoder to embed the full graph context.

A decoder reconstructs the original features of masked nodes only.

This self-supervised signal enables pretraining without labels.
"""

import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv
import random
 
# 1. Load dataset
dataset = Planetoid(root='/tmp/Cora', name='Cora')
data = dataset[0]
 
# 2. Define GCN encoder and attribute decoder
class GCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
 
    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        return self.conv2(x, edge_index)
 
class AttributeDecoder(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        self.linear = torch.nn.Linear(hidden_channels, out_channels)
 
    def forward(self, h):
        return self.linear(h)
 
# 3. Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
encoder = GCNEncoder(dataset.num_node_features, 64).to(device)
decoder = AttributeDecoder(64, dataset.num_node_features).to(device)
optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=0.01)
loss_fn = torch.nn.MSELoss()
 
# 4. Mask node features (BERT-style masking)
def mask_node_features(x, mask_ratio=0.3):
    x = x.clone()
    num_nodes = x.size(0)
    num_mask = int(mask_ratio * num_nodes)
    mask_indices = random.sample(range(num_nodes), num_mask)
    masked_x = x.clone()
    masked_x[mask_indices] = 0  # zero-out masked features
    return masked_x, torch.tensor(mask_indices, dtype=torch.long)
 
# 5. Training loop
data = data.to(device)
for epoch in range(1, 101):
    encoder.train()
    decoder.train()
    optimizer.zero_grad()
 
    masked_x, mask_idx = mask_node_features(data.x)
    z = encoder(masked_x.to(device), data.edge_index)
    recon = decoder(z[mask_idx])
    target = data.x[mask_idx]
 
    loss = loss_fn(recon, target)
    loss.backward()
    optimizer.step()
 
    if epoch % 10 == 0:
        print(f"Epoch {epoch:03d}, Masking Loss: {loss.item():.4f}")