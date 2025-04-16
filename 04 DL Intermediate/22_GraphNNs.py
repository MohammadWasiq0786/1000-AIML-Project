"""
Project 142. Graph neural networks
Description:
Graph Neural Networks (GNNs) extend deep learning to graph-structured data. They learn representations by aggregating features from neighboring nodes, making them ideal for tasks like node classification, link prediction, and graph classification. In this project, we implement a GNN for node classification on a citation network using PyTorch Geometric (PyG).

About:
🧠 What This Project Demonstrates:
Implements a Graph Convolutional Network (GCN)

Performs node classification on the Cora citation network

Aggregates neighbor features using message passing on graph structure
"""

# Install required packages:
# pip install torch torchvision torch-geometric torch-scatter torch-sparse torch-cluster torch-spline-conv
 
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv
import matplotlib.pyplot as plt
 
# Load Cora dataset
dataset = Planetoid(root='data/Cora', name='Cora')
data = dataset[0]  # Single graph
 
# Define GCN model
class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
 
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        return self.conv2(x, edge_index)
 
# Instantiate model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GCN(dataset.num_node_features, 16, dataset.num_classes).to(device)
data = data.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
 
# Training loop
def train():
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()
 
# Evaluation
def test():
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)
    accs = []
    for mask in [data.train_mask, data.val_mask, data.test_mask]:
        correct = pred[mask].eq(data.y[mask]).sum().item()
        accs.append(correct / mask.sum().item())
    return accs
 
# Train for a few epochs
for epoch in range(1, 201):
    loss = train()
    train_acc, val_acc, test_acc = test()
    if epoch % 20 == 0:
        print(f"Epoch {epoch:3d} | Loss: {loss:.4f} | Train: {train_acc:.2f} | Val: {val_acc:.2f} | Test: {test_acc:.2f}")