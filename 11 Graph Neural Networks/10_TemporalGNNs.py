"""
Project 410. Temporal graph neural networks
Description:
Temporal Graph Neural Networks handle graphs where the structure and features evolve over time — such as in transaction networks, communication graphs, or social feeds. Unlike static GNNs, TGNNs model sequential dynamics using techniques like RNNs, attention, or time encodings. In this project, we’ll build a basic TGNN using PyTorch Geometric Temporal, applying it to a dynamic node classification task.

About:
✅ What It Does:
Loads a real-world temporal graph of Bitcoin transactions.

Implements EvolveGCNH, a recurrent GNN that updates node states over time.

Performs dynamic node classification (e.g., fraud detection over time).
"""

import torch
import torch.nn.functional as F
from torch_geometric_temporal.dataset import EllipticBitcoinDataset
from torch_geometric_temporal.nn.recurrent import EvolveGCNH
from tqdm import tqdm
 
# 1. Load dynamic graph dataset
dataset = EllipticBitcoinDataset()
loader = dataset.get_dataset()
snapshot_iterator = iter(loader)
 
# 2. Define the Temporal GNN model
class RecurrentGCN(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.recurrent = EvolveGCNH(in_channels, out_channels)
        self.linear = torch.nn.Linear(out_channels, 2)  # Binary classification
 
    def forward(self, x, edge_index, edge_weight, h):
        h = self.recurrent(x, edge_index, edge_weight, h)
        return self.linear(h), h
 
# 3. Model, optimizer, setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = RecurrentGCN(in_channels=dataset.num_node_features, out_channels=32).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = torch.nn.CrossEntropyLoss()
 
# 4. Train model over time snapshots
model.train()
h = None  # Initial hidden state
 
for epoch in range(1, 11):  # 10 epochs
    snapshot_iterator = iter(loader)
    total_loss = 0
    for snapshot in snapshot_iterator:
        x = snapshot.x.to(device)
        y = snapshot.y.to(device)
        edge_index = snapshot.edge_index.to(device)
        edge_weight = snapshot.edge_attr.to(device) if snapshot.edge_attr is not None else None
 
        optimizer.zero_grad()
        out, h = model(x, edge_index, edge_weight, h)
        loss = loss_fn(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
 
    print(f"Epoch {epoch:02d}, Total Loss: {total_loss:.4f}")