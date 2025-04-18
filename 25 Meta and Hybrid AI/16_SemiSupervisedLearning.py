"""
Project 976: Semi-supervised Learning Implementation
Description
Semi-supervised learning is a machine learning approach where the model is trained on a small amount of labeled data and a large amount of unlabeled data. This project demonstrates the use of semi-supervised learning using consistency regularization to make predictions on unlabeled data more reliable and improve generalization.

Key Concepts Covered:
Pseudo-Labeling: A technique for semi-supervised learning where the model generates labels for unlabeled data, which are then treated as ground truth during training.

Semi-supervised Learning: A method of training models using a small amount of labeled data and a large amount of unlabeled data.

Self-training: The model iteratively learns from its own predictions on the unlabeled data.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import random
 
# Define a simple neural network for CIFAR-10 classification
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.fc1 = nn.Linear(64 * 6 * 6, 128)
        self.fc2 = nn.Linear(128, 10)
 
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
 
# Load CIFAR-10 dataset
transform = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()])
train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
 
# Split the dataset into labeled and unlabeled sets
labeled_indices = random.sample(range(len(train_dataset)), 1000)  # Start with 1000 labeled samples
unlabeled_indices = list(set(range(len(train_dataset))) - set(labeled_indices))
 
labeled_data = Subset(train_dataset, labeled_indices)
unlabeled_data = Subset(train_dataset, unlabeled_indices)
 
# Create DataLoaders for labeled and unlabeled data
labeled_loader = DataLoader(labeled_data, batch_size=32, shuffle=True)
unlabeled_loader = DataLoader(unlabeled_data, batch_size=32, shuffle=False)
 
# Initialize the model, optimizer, and loss function
model = SimpleCNN()
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()
 
# Semi-supervised learning loop using Pseudo-Labeling
for epoch in range(5):  # Semi-supervised training for 5 epochs
    model.train()
    total_loss = 0
 
    # Train on labeled data
    for data, target in labeled_loader:
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
 
    # Pseudo-labeling: Assign labels to the unlabeled data
    model.eval()
    pseudo_labels = []
    with torch.no_grad():
        for data, _ in unlabeled_loader:
            output = model(data)
            pseudo_label = torch.argmax(output, dim=1)
            pseudo_labels.append(pseudo_label)
 
    # Update the labeled dataset with pseudo-labeled data
    pseudo_labels = torch.cat(pseudo_labels, dim=0)
    pseudo_labeled_data = [(unlabeled_data[i][0], pseudo_labels[i]) for i in range(len(unlabeled_data))]
    
    # Add the pseudo-labeled data to the training loop
    augmented_loader = DataLoader(labeled_data + pseudo_labeled_data, batch_size=32, shuffle=True)
 
    # Train the model on both labeled and pseudo-labeled data
    for data, target in augmented_loader:
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
 
    print(f"Epoch {epoch+1}, Loss: {total_loss / len(labeled_loader)}")
 
# Evaluate the model on the test dataset
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for data, target in DataLoader(test_dataset, batch_size=32, shuffle=False):
        output = model(data)
        _, predicted = torch.max(output, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()
 
print(f"Test Accuracy: {100 * correct / total:.2f}%")