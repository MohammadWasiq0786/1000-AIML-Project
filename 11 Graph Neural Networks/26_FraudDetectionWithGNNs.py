"""
Project 426. Fraud detection with GNNs
Description:
Fraud detection often involves identifying suspicious activity in transaction networks, where entities (like users, accounts) and interactions (like money transfers) form a graph. Graph Neural Networks (GNNs) can capture complex inter-entity relationships and detect fraud by learning structural patterns. In this project, we’ll use a GCN model on a synthetic transaction graph to classify nodes as fraudulent or not.

About:
✅ What It Does:
Simulates a transaction graph using preferential attachment (realistic fraud topology).

Labels a few nodes as fraudulent and trains a GCN to detect them.

Uses random features (replace with real transaction features in production).

Evaluates classification accuracy on unseen nodes.
"""

import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import networkx as nx
import matplotlib.pyplot as plt
import random
 
# 1. Simulate a small transaction network
G = nx.barabasi_albert_graph(50, 2)  # 50 accounts, preferential attachment
 
# Simulate fraud labels (10% fraudulent)
labels = torch.zeros(50, dtype=torch.long)
fraud_nodes = random.sample(range(50), 5)
labels[fraud_nodes] = 1
 
# Convert graph to edge index
edge_index = torch.tensor(list(G.edges), dtype=torch.long).t().contiguous()
edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)  # make undirected
 
# Node features: random (or transaction stats if real)
x = torch.rand((50, 16))
 
# Train/Test mask
train_mask = torch.zeros(50, dtype=torch.bool)
test_mask = torch.zeros(50, dtype=torch.bool)
train_mask[:35] = True
test_mask[35:] = True
 
# 2. Create data object
data = Data(x=x, edge_index=edge_index, y=labels, train_mask=train_mask, test_mask=test_mask)
 
# 3. Define GCN model
class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
 
    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.3, training=self.training)
        return self.conv2(x, edge_index)
 
# 4. Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GCN(16, 32, 2).to(device)
data = data.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = torch.nn.CrossEntropyLoss()
 
# 5. Train function
def train():
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = loss_fn(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()
 
# 6. Test function
def test():
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)
    correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
    total = data.test_mask.sum()
    return int(correct) / int(total)
 
# 7. Run training loop
for epoch in range(1, 31):
    loss = train()
    acc = test()
    print(f"Epoch {epoch:02d}, Loss: {loss:.4f}, Test Accuracy: {acc:.4f}")