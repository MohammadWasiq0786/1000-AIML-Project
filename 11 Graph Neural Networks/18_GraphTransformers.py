"""
Project 418. Graph transformers implementation
Description:
Graph Transformers combine the power of self-attention with the inductive biases of graph structures. Unlike traditional GNNs that rely purely on local neighborhoods, transformers can capture long-range dependencies in graphs using attention over node pairs. In this project, we’ll implement a Graph Transformer using PyTorch Geometric on the ZINC dataset for graph regression.

About:
✅ What It Does:
Uses Graph Transformer to model complex interactions over molecular graphs.

Applies self-attention across graph nodes, capturing both local and global patterns.

Trains on ZINC for molecular property prediction (regression).
"""

# pip install torch-geometric

import torch
import torch.nn.functional as F
from torch_geometric.datasets import ZINC
from torch_geometric.nn import GraphTransformer, global_mean_pool
from torch_geometric.loader import DataLoader
 
# 1. Load ZINC molecular graph dataset
train_dataset = ZINC(root='/tmp/ZINC', split='train')
val_dataset = ZINC(root='/tmp/ZINC', split='val')
test_dataset = ZINC(root='/tmp/ZINC', split='test')
 
train_loader = DataLoader(train_dataset[:1000], batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset[:200], batch_size=32)
test_loader = DataLoader(test_dataset[:200], batch_size=32)
 
# 2. Define Graph Transformer model
class GraphTransformerNet(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.transformer = GraphTransformer(in_channels=in_channels, hidden_channels=hidden_channels,
                                            num_layers=3, heads=4, dropout=0.1)
        self.lin1 = torch.nn.Linear(hidden_channels, hidden_channels)
        self.lin2 = torch.nn.Linear(hidden_channels, 1)
 
    def forward(self, x, edge_index, edge_attr, batch):
        x = self.transformer(x, edge_index, edge_attr)
        x = global_mean_pool(x, batch)
        x = F.relu(self.lin1(x))
        return self.lin2(x).squeeze()
 
# 3. Model setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GraphTransformerNet(in_channels=train_dataset.num_node_features, hidden_channels=64).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = torch.nn.L1Loss()  # MAE
 
# 4. Training function
def train():
    model.train()
    total_loss = 0
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        loss = loss_fn(pred, batch.y.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)
 
# 5. Validation
def evaluate(loader):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            loss = loss_fn(pred, batch.y.view(-1))
            total_loss += loss.item()
    return total_loss / len(loader)
 
# 6. Run training loop
for epoch in range(1, 21):
    train_loss = train()
    val_loss = evaluate(val_loader)
    print(f"Epoch {epoch:02d}, Train MAE: {train_loss:.4f}, Val MAE: {val_loss:.4f}")