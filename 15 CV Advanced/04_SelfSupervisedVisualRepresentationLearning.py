"""
Project 564: Self-supervised Visual Representation Learning
Description:
Self-supervised learning in computer vision aims to learn useful representations of data without relying on labeled data. In this project, we will implement a self-supervised learning approach for visual representation learning, using models such as SimCLR or MoCo. These models learn representations by predicting context or augmentations of the input image without using labeled data.
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import datasets, models
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
 
# 1. Define the SimCLR model (a simplified version)
class SimCLR(nn.Module):
    def __init__(self, base_model):
        super(SimCLR, self).__init__()
        self.base_model = base_model
        self.fc = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(512, 128)
        )
 
    def forward(self, x):
        x = self.base_model(x)
        x = self.fc(x)
        return x
 
# 2. Load a pre-trained ResNet model and use it as the base model for SimCLR
resnet = models.resnet18(pretrained=True)
resnet.fc = nn.Identity()  # Remove the final classification layer
simclr_model = SimCLR(resnet)
 
# 3. Set up dataset and data loaders with augmentation (for self-supervised learning)
transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
 
train_dataset = CIFAR10(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
 
# 4. Define contrastive loss function (NT-Xent Loss)
def contrastive_loss(x1, x2, temperature=0.07):
    cosine_similarity = nn.functional.cosine_similarity(x1.unsqueeze(1), x2.unsqueeze(0), dim=-1)
    labels = torch.arange(x1.size(0)).long().to(x1.device)
    logits = cosine_similarity / temperature
    loss = nn.CrossEntropyLoss()(logits, labels)
    return loss
 
# 5. Train the model (simplified version)
optimizer = torch.optim.Adam(simclr_model.parameters(), lr=0.001)
 
# Simulated training loop
for epoch in range(10):  # For simplicity, run for 10 epochs
    simclr_model.train()
    total_loss = 0
    for images, _ in train_loader:
        optimizer.zero_grad()
 
        # Generate augmented views (here, using the same batch for simplicity)
        x1, x2 = images, images  # In practice, augment the images
 
        # Forward pass through SimCLR
        z1, z2 = simclr_model(x1), simclr_model(x2)
 
        # Compute the contrastive loss
        loss = contrastive_loss(z1, z2)
        loss.backward()
        optimizer.step()
 
        total_loss += loss.item()
 
    print(f"Epoch {epoch+1}, Loss: {total_loss / len(train_loader)}")