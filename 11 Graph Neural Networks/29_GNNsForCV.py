"""
Project 429. Graph neural networks for computer vision
Description:
In computer vision, images can be transformed into graphs — with pixels, superpixels, or regions as nodes and spatial or semantic relationships as edges. This allows Graph Neural Networks (GNNs) to capture non-Euclidean structure and context-aware reasoning beyond traditional CNNs. In this project, we’ll use a superpixel-based graph for image classification with a GCN.

About:
✅ What It Does:
Converts CIFAR10 images into superpixel graphs.

Uses GCN layers to classify the image based on graph structure.

Aggregates features using global mean pooling.

Evaluates graph-based image classification performance.
"""

# pip install torch-geometric torchvision

import torch
import torch.nn.functional as F
from torch_geometric.datasets import SuperpixelsSLIC
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.loader import DataLoader
 
# 1. Load CIFAR10 Superpixel dataset (each image is a graph)
dataset = SuperpixelsSLIC(root='/tmp/superpixels', name='CIFAR10', split='train', transform=None)
dataset = dataset.shuffle()
train_dataset = dataset[:500]
test_dataset = dataset[500:600]
 
train_loader = DataLoader(train_dataset, batch_size=32)
test_loader = DataLoader(test_dataset, batch_size=32)
 
# 2. Define GCN model for superpixel graph classification
class SuperpixelGCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_classes):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.lin1 = torch.nn.Linear(hidden_channels, 64)
        self.lin2 = torch.nn.Linear(64, num_classes)
 
    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = global_mean_pool(x, batch)
        x = F.relu(self.lin1(x))
        return F.log_softmax(self.lin2(x), dim=1)
 
# 3. Model setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SuperpixelGCN(in_channels=dataset.num_node_features, hidden_channels=64, num_classes=10).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = torch.nn.NLLLoss()
 
# 4. Train function
def train():
    model.train()
    total_loss = 0
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index, batch.batch)
        loss = loss_fn(out, batch.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)
 
# 5. Evaluate function
def test():
    model.eval()
    correct = 0
    for batch in test_loader:
        batch = batch.to(device)
        out = model(batch.x, batch.edge_index, batch.batch)
        pred = out.argmax(dim=1)
        correct += (pred == batch.y).sum().item()
    return correct / len(test_dataset)
 
# 6. Training loop
for epoch in range(1, 21):
    loss = train()
    acc = test()
    print(f"Epoch {epoch:02d}, Loss: {loss:.4f}, Test Acc: {acc:.4f}")