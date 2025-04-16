"""
Project 434. GNNs for social good applications
Description:
Graph Neural Networks can be powerful tools for addressing social good problems like disease spread modeling, poverty prediction, disaster response coordination, or education access mapping. In this project, we’ll simulate a social contact network (e.g., during a pandemic) and use a GCN to predict risk levels of individuals based on their connections — a simplified model for contagion or vulnerability prediction.

About:
✅ What It Does:
Simulates a social contact graph with synthetic features and risk labels.

Trains a GCN model to predict which individuals are high-risk.

Demonstrates how GNNs can be used for pandemic response, disaster risk mapping, or health vulnerability prediction.
"""

# pip install torch-geometric networkx

import torch
import torch.nn.functional as F
import networkx as nx
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import random
 
# 1. Generate a synthetic social contact graph
G = nx.barabasi_albert_graph(n=100, m=3)
edge_index = torch.tensor(list(G.edges), dtype=torch.long).t().contiguous()
edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)  # undirected
 
# 2. Node features (e.g., age, exposure level) — random for now
x = torch.rand((100, 4))
 
# 3. Simulate risk labels (1 = high-risk, 0 = low-risk)
labels = torch.zeros(100, dtype=torch.long)
high_risk = random.sample(range(100), 20)
labels[high_risk] = 1
 
# 4. Train/test split
train_mask = torch.zeros(100, dtype=torch.bool)
test_mask = torch.zeros(100, dtype=torch.bool)
train_mask[:70] = True
test_mask[70:] = True
 
# 5. Create graph data object
data = Data(x=x, edge_index=edge_index, y=labels, train_mask=train_mask, test_mask=test_mask)
 
# 6. GCN for node classification
class RiskGCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(4, 16)
        self.conv2 = GCNConv(16, 2)
 
    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.3, training=self.training)
        return self.conv2(x, edge_index)
 
# 7. Training setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = RiskGCN().to(device)
data = data.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = torch.nn.CrossEntropyLoss()
 
# 8. Training function
def train():
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = loss_fn(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()
 
# 9. Test function
def test():
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)
    correct = int((pred[data.test_mask] == data.y[data.test_mask]).sum())
    total = int(data.test_mask.sum())
    return correct / total
 
# 10. Run training loop
for epoch in range(1, 31):
    loss = train()
    acc = test()
    print(f"Epoch {epoch:02d}, Loss: {loss:.4f}, Test Accuracy: {acc:.4f}")