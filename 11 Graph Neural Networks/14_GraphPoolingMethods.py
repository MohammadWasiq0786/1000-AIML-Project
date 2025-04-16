"""
Project 414. Graph pooling methods
Description:
Graph pooling reduces the size of a graph while preserving its structure and information — similar to pooling in CNNs. It’s critical in graph classification and hierarchical learning. There are several pooling methods like TopKPooling, SAGPool, and global pooling (mean/max). In this project, we’ll implement global pooling and TopK pooling for graph classification using the MUTAG dataset.

About:
✅ What It Does:
Uses GCN layers for feature learning.

Applies TopKPooling to reduce the graph size while keeping the most important nodes.

Combines pooled representations with global_mean_pool for graph-level embedding.

Performs graph classification on MUTAG (e.g., molecule classification).
"""

# pip install torch-geometric


import torch
import torch.nn.functional as F
from torch_geometric.datasets import TUDataset
from torch_geometric.nn import GCNConv, global_mean_pool, TopKPooling
from torch_geometric.loader import DataLoader
 
# 1. Load MUTAG graph classification dataset
dataset = TUDataset(root='/tmp/MUTAG', name='MUTAG')
loader = DataLoader(dataset, batch_size=32, shuffle=True)
 
# 2. Define a GCN with TopK pooling
class GCNWithPooling(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(dataset.num_node_features, 64)
        self.pool1 = TopKPooling(64, ratio=0.8)
        self.conv2 = GCNConv(64, 64)
        self.pool2 = TopKPooling(64, ratio=0.8)
        self.linear1 = torch.nn.Linear(64*2, 64)
        self.linear2 = torch.nn.Linear(64, dataset.num_classes)
 
    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x, edge_index, _, batch, _, _ = self.pool1(x, edge_index, batch=batch)
        x1 = global_mean_pool(x, batch)
 
        x = F.relu(self.conv2(x, edge_index))
        x, edge_index, _, batch, _, _ = self.pool2(x, edge_index, batch=batch)
        x2 = global_mean_pool(x, batch)
 
        x = torch.cat([x1, x2], dim=1)
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return F.log_softmax(x, dim=1)
 
# 3. Setup model, optimizer, loss
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GCNWithPooling().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = torch.nn.NLLLoss()
 
# 4. Training function
def train():
    model.train()
    total_loss = 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index, batch.batch)
        loss = loss_fn(out, batch.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)
 
# 5. Train the model
for epoch in range(1, 31):
    loss = train()
    print(f"Epoch {epoch:02d}, Loss: {loss:.4f}")