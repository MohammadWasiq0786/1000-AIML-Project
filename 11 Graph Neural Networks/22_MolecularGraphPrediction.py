"""
Project 422. Molecular graph prediction
Description:
In molecular graph prediction, each molecule is represented as a graph where atoms are nodes and bonds are edges. The goal is to predict chemical properties like toxicity, solubility, or bioactivity. Graph Neural Networks (GNNs) excel at this because they respect molecular structure. In this project, we’ll use a GCN model to predict a molecular property from the Tox21 dataset.

About:
✅ What It Does:
Loads the Tox21 dataset where each molecule is a graph.

Trains a GCN model to predict multiple chemical properties.

Uses binary cross-entropy loss and sigmoid for multi-label classification.

Handles missing labels via masking.
"""


# pip install torch-geometric

import torch
import torch.nn.functional as F
from torch_geometric.datasets import MoleculeNet
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.loader import DataLoader
 
# 1. Load molecular dataset
dataset = MoleculeNet(root='/tmp/Tox21', name='Tox21')
dataset = dataset.shuffle()
train_dataset = dataset[:500]
test_dataset = dataset[500:700]
train_loader = DataLoader(train_dataset, batch_size=32)
test_loader = DataLoader(test_dataset, batch_size=32)
 
# 2. Define GCN model for graph regression
class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(dataset.num_node_features, 64)
        self.conv2 = GCNConv(64, 64)
        self.fc1 = torch.nn.Linear(64, 32)
        self.fc2 = torch.nn.Linear(32, dataset.num_tasks)  # multi-task binary prediction
 
    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = global_mean_pool(x, batch)
        x = F.relu(self.fc1(x))
        return torch.sigmoid(self.fc2(x))  # multi-label binary classification
 
# 3. Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GCN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = torch.nn.BCELoss()
 
# 4. Training loop
def train():
    model.train()
    total_loss = 0
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        pred = model(batch.x, batch.edge_index, batch.batch)
        # Handle missing labels by masking
        mask = ~torch.isnan(batch.y)
        loss = loss_fn(pred[mask], batch.y[mask])
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)
 
# 5. Evaluation
def test():
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            pred = model(batch.x, batch.edge_index, batch.batch)
            mask = ~torch.isnan(batch.y)
            pred_binary = (pred > 0.5).float()
            correct += (pred_binary[mask] == batch.y[mask]).sum().item()
            total += mask.sum().item()
    return correct / total
 
# 6. Run training
for epoch in range(1, 21):
    loss = train()
    acc = test()
    print(f"Epoch {epoch:02d}, Loss: {loss:.4f}, Test Accuracy: {acc:.4f}")