"""
Project 150. Few-shot learning implementation
Description:
Few-shot learning enables models to recognize new classes with just a few labeled examples per class. It simulates human-like learning where a model can generalize from limited data. This project implements a prototypical network, a popular few-shot learning approach, on a toy dataset using PyTorch.

About:
🧠 What This Project Demonstrates:
Simulates few-shot learning episodes with n-way k-shot tasks

Implements a Prototypical Network for learning from few examples

Learns class prototypes and classifies by distance to them
"""

# Install if not already: pip install torch torchvision matplotlib
 
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np
import random
 
# Preprocessing
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
dataset = MNIST(root='./data', train=True, transform=transform, download=True)
 
# Few-shot episode creator
def create_episode(dataset, n_way=3, k_shot=5, q_queries=5):
    classes = np.random.choice(list(set(dataset.targets.numpy())), n_way, replace=False)
    support_images, support_labels = [], []
    query_images, query_labels = [], []
 
    for i, cls in enumerate(classes):
        idxs = (dataset.targets == cls).nonzero(as_tuple=True)[0]
        selected = idxs[torch.randperm(len(idxs))][:k_shot + q_queries]
        support = selected[:k_shot]
        query = selected[k_shot:]
 
        support_images.append(dataset.data[support])
        query_images.append(dataset.data[query])
        support_labels.extend([i] * k_shot)
        query_labels.extend([i] * q_queries)
 
    support_images = torch.cat(support_images).unsqueeze(1).float() / 255
    query_images = torch.cat(query_images).unsqueeze(1).float() / 255
    support_labels = torch.tensor(support_labels)
    query_labels = torch.tensor(query_labels)
    return support_images, support_labels, query_images, query_labels
 
# Embedding network
class EmbeddingNet(nn.Module):
    def __init__(self):
        super(EmbeddingNet, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3), nn.ReLU(), nn.MaxPool2d(2)
        )
    def forward(self, x):
        return self.net(x).view(x.size(0), -1)  # Flatten
 
# Prototypical Network Logic
def prototypical_loss(support_emb, support_lbl, query_emb, query_lbl, n_way):
    # Compute prototypes (mean embeddings for each class)
    prototypes = torch.stack([
        support_emb[support_lbl == i].mean(0)
        for i in range(n_way)
    ])
 
    # Compute distances to each prototype
    dists = torch.cdist(query_emb, prototypes)
    pred = torch.argmin(dists, dim=1)
    acc = (pred == query_lbl).float().mean().item()
    return F.cross_entropy(-dists, query_lbl), acc
 
# Run a single episode
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EmbeddingNet().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
 
support_x, support_y, query_x, query_y = create_episode(dataset)
support_x, query_x = support_x.to(device), query_x.to(device)
support_y, query_y = support_y.to(device), query_y.to(device)
 
model.train()
support_emb = model(support_x)
query_emb = model(query_x)
 
loss, acc = prototypical_loss(support_emb, support_y, query_emb, query_y, n_way=3)
optimizer.zero_grad()
loss.backward()
optimizer.step()
 
print(f"🔍 Few-Shot Episode — Loss: {loss.item():.4f}, Accuracy: {acc * 100:.2f}%")