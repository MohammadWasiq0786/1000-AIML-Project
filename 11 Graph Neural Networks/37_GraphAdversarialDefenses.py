"""
Project 437. Graph adversarial defenses
Description:
While GNNs are powerful, they can be vulnerable to adversarial attacks (e.g., fake edges or malicious nodes). Defenses aim to make models more robust through strategies like graph purification, adversarial training, or robust aggregation functions. In this project, we implement a basic defense via edge dropout, which makes the model more resilient by randomly dropping edges during training.

About:
✅ What It Does:
Adds edge dropout during training (drop 20% edges at random).

Trains a robust GCN model that learns to be less dependent on any single edge.

Helps mitigate effects of adversarial edge manipulations.

Can be extended with other defenses like adversarial training or attention filtering.
"""

import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv
from torch_geometric.utils import dropout_edge
 
# 1. Load Cora dataset
dataset = Planetoid(root='/tmp/Cora', name='Cora')
data = dataset[0]
 
# 2. GCN with dropout-enabled training
class RobustGCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
 
    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.3, training=self.training)
        return self.conv2(x, edge_index)
 
# 3. Model setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = RobustGCN(dataset.num_node_features, 16, dataset.num_classes).to(device)
data = data.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = torch.nn.CrossEntropyLoss()
 
# 4. Training loop with edge dropout defense
def train():
    model.train()
    optimizer.zero_grad()
    dropped_edge_index, _ = dropout_edge(data.edge_index, p=0.2)  # randomly drop 20% edges
    out = model(data.x, dropped_edge_index)
    loss = loss_fn(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()
 
# 5. Evaluation without edge dropout
def test():
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)
    correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
    return int(correct) / int(data.test_mask.sum())
 
# 6. Run training
for epoch in range(1, 51):
    loss = train()
    acc = test()
    print(f"Epoch {epoch:02d}, Loss: {loss:.4f}, Test Accuracy: {acc:.4f}")