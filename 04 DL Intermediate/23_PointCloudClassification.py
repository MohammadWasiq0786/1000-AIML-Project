"""
Project 143. Point cloud classification
Description:
Point cloud classification involves processing unordered sets of 3D coordinates (points in space) to recognize object categories (e.g., car, chair, airplane). This project uses PointNet, a deep learning architecture designed to handle unordered point sets directly and perform classification on 3D point cloud data.

About:
🧠 What This Project Demonstrates:
Implements a simplified PointNet-like network

Processes raw 3D point cloud data without voxelization or meshing

Uses global max pooling to get permutation-invariant features
"""

# Install dependencies:
# pip install torch torchvision torch-geometric pytorch3d numpy
 
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import ModelNet
from torch_geometric.transforms import SamplePoints
from torch_geometric.loader import DataLoader
 
# Load ModelNet10 (3D objects), sample 1024 points per mesh
transform = SamplePoints(1024)
train_dataset = ModelNet(root='data/ModelNet10', name='10', train=True, transform=transform)
test_dataset = ModelNet(root='data/ModelNet10', name='10', train=False, transform=transform)
 
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16)
 
# Define a simplified PointNet model
class SimplePointNet(nn.Module):
    def __init__(self, num_classes):
        super(SimplePointNet, self).__init__()
        self.fc1 = nn.Linear(3, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 1024)
        self.fc4 = nn.Linear(1024, 512)
        self.fc5 = nn.Linear(512, 256)
        self.fc6 = nn.Linear(256, num_classes)
 
    def forward(self, data):
        x = data.pos.view(data.pos.size(0), -1, 3)  # (B, N, 3)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))  # (B, N, 1024)
        x = torch.max(x, 1)[0]   # Global max pooling (B, 1024)
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        return self.fc6(x)
 
# Initialize model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimplePointNet(num_classes=10).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()
 
# Training loop (one epoch for demo)
model.train()
for data in train_loader:
    data = data.to(device)
    optimizer.zero_grad()
    output = model(data)
    loss = criterion(output, data.y)
    loss.backward()
    optimizer.step()
    print(f"🧠 Training Loss: {loss.item():.4f}")
    break