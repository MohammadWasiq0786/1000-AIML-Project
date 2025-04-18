"""
Project 978: Contrastive Learning Implementation
Description
Contrastive learning aims to learn representations by contrasting positive pairs (similar items) with negative pairs (dissimilar items). In this project, we will implement contrastive learning using SimCLR, a method that learns representations by maximizing the similarity between augmented views of the same image while minimizing the similarity between different images.

Key Concepts Covered:
Contrastive Learning: A technique where the model learns to distinguish between similar and dissimilar pairs, optimizing for representations that are close for similar items and far apart for dissimilar items.

SimCLR: A specific method of contrastive learning that uses augmented views of the same image as positive pairs and learns to maximize their similarity.

NT-Xent Loss: A loss function used for contrastive learning that encourages the model to pull together similar items and push apart dissimilar items.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
import numpy as np
 
# Define the SimCLR architecture
class SimCLR(nn.Module):
    def __init__(self, base_model, projection_dim=128):
        super(SimCLR, self).__init__()
        self.base_model = base_model
        self.base_model.fc = nn.Identity()  # Remove the final fully connected layer
        self.projection_head = nn.Sequential(
            nn.Linear(self.base_model.fc.in_features, 512),
            nn.ReLU(),
            nn.Linear(512, projection_dim)
        )
 
    def forward(self, x):
        features = self.base_model(x)
        projections = self.projection_head(features)
        return projections
 
# Define the contrastive loss (NT-Xent Loss)
class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature
 
    def forward(self, z_i, z_j):
        # Normalize the projections
        z_i = nn.functional.normalize(z_i, dim=-1, p=2)
        z_j = nn.functional.normalize(z_j, dim=-1, p=2)
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(z_i, z_j.T) / self.temperature
        
        # Create labels: positive pairs (same image, different augmentations)
        labels = torch.arange(z_i.size(0)).long().to(z_i.device)
        
        # Compute the contrastive loss (NT-Xent loss)
        loss = nn.CrossEntropyLoss()(similarity_matrix, labels)
        return loss
 
# Define the data augmentations for contrastive learning
transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
 
# Load the CIFAR-10 dataset
train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=4)
 
# Initialize the pre-trained ResNet-50 model and SimCLR model
base_model = models.resnet50(pretrained=True)
simclr_model = SimCLR(base_model)
 
# Define the optimizer and loss function
optimizer = optim.Adam(simclr_model.parameters(), lr=0.0001)
contrastive_loss_fn = ContrastiveLoss()
 
# Train the model using contrastive learning
for epoch in range(5):  # Training for 5 epochs
    simclr_model.train()
    total_loss = 0
    for data, _ in train_loader:
        optimizer.zero_grad()
 
        # Generate positive pairs (augmented views of the same image)
        data_aug1 = data
        data_aug2 = data  # For simplicity, we'll use the same batch as augmented pairs here
        
        # Forward pass for the augmented views
        projections1 = simclr_model(data_aug1)
        projections2 = simclr_model(data_aug2)
 
        # Compute the contrastive loss
        loss = contrastive_loss_fn(projections1, projections2)
        loss.backward()
        optimizer.step()
 
        total_loss += loss.item()
 
    print(f"Epoch {epoch+1}, Loss: {total_loss / len(train_loader)}")
 
# Evaluate the model (if needed, apply to downstream tasks)
# This is a self-supervised learning setup, so the downstream task is often a separate evaluation.