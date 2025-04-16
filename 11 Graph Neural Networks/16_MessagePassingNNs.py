"""
Project 416. Message passing neural networks
Description:
Message Passing Neural Networks (MPNNs) are a general framework for GNNs where nodes update their states by receiving messages from neighbors, followed by aggregation and update steps. MPNNs are powerful and flexible, making them suitable for molecular property prediction, social dynamics, and knowledge graphs. In this project, we’ll implement an MPNN using PyTorch Geometric for a regression task on the QM9 molecular dataset.

About:
✅ What It Does:
Uses NNConv for edge-conditioned message passing.

Uses a GRU to iteratively update node representations.

Aggregates node info with Set2Set for graph-level output.

Trains on the QM9 molecular dataset to predict chemical properties.
"""

import torch
import torch.nn.functional as F
from torch_geometric.datasets import QM9
from torch_geometric.nn import NNConv, Set2Set
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.loader import DataLoader
 
# 1. Load QM9 dataset
target = 0  # We'll predict the first target property
dataset = QM9(root='/tmp/QM9')
dataset = dataset.shuffle()
dataset = dataset[:1000]  # Keep it light for demo
 
# 2. Define MPNN model
class MPNN(torch.nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_dim):
        super().__init__()
        nn = Sequential(Linear(edge_dim, 128), ReLU(), Linear(128, node_dim * hidden_dim))
        self.nnconv = NNConv(node_dim, hidden_dim, nn, aggr='mean')
        self.gru = torch.nn.GRU(hidden_dim, hidden_dim)
        self.set2set = Set2Set(hidden_dim, processing_steps=3)
        self.fc1 = Linear(2 * hidden_dim, hidden_dim)
        self.fc2 = Linear(hidden_dim, 1)
 
    def forward(self, data):
        out = F.relu(self.nnconv(data.x, data.edge_index, data.edge_attr))
        h = out.unsqueeze(0)
        for _ in range(3):
            out, h = self.gru(out.unsqueeze(0), h)
            out = out.squeeze(0)
        out = self.set2set(out, data.batch)
        out = F.relu(self.fc1(out))
        return self.fc2(out).squeeze()
 
# 3. Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MPNN(node_dim=dataset.num_node_features, edge_dim=dataset.num_edge_features, hidden_dim=64).to(device)
loader = DataLoader(dataset, batch_size=32)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = torch.nn.L1Loss()  # MAE
 
# 4. Training loop
def train():
    model.train()
    total_loss = 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        pred = model(batch)
        loss = loss_fn(pred, batch.y[:, target])
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)
 
# 5. Run training
for epoch in range(1, 21):
    loss = train()
    print(f"Epoch {epoch:02d}, MAE Loss: {loss:.4f}")