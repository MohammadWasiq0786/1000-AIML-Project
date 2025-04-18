"""
Project 975: Active Learning System
Description
Active learning is a machine learning paradigm where the model actively selects the most informative examples to be labeled, typically from a pool of unlabeled data. This can significantly reduce the amount of labeled data required for training. In this project, we will implement a simple active learning system using uncertainty sampling.

Key Concepts Covered:
Active Learning: The model selects the most informative examples to be labeled by querying an oracle (e.g., human annotators).

Uncertainty Sampling: A common strategy in active learning, where the model queries examples it is least confident about (i.e., those near the decision boundary).

Curriculum Learning: While not directly implemented here, active learning can be seen as a curriculum-like strategy for efficiently learning with less data.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset, Subset
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import random
 
# Define a simple CNN for active learning demonstration
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
 
# Uncertainty Sampling for Active Learning
def uncertainty_sampling(model, unlabeled_data, n_samples=10):
    model.eval()
    uncertainties = []
    for data, _ in unlabeled_data:
        output = model(data.unsqueeze(0))  # Make prediction
        probs = torch.softmax(output, dim=1)
        uncertainty = -torch.max(probs).item()  # Calculate uncertainty (1 - max probability)
        uncertainties.append(uncertainty)
    
    # Select samples with the highest uncertainty
    uncertain_indices = np.argsort(uncertainties)[-n_samples:]
    return uncertain_indices
 
# Load CIFAR-10 dataset
transform = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()])
dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
 
# Split into labeled and unlabeled sets
labeled_indices = random.sample(range(len(dataset)), 100)  # Start with 100 labeled samples
unlabeled_indices = list(set(range(len(dataset))) - set(labeled_indices))
 
labeled_data = torch.utils.data.Subset(dataset, labeled_indices)
unlabeled_data = torch.utils.data.Subset(dataset, unlabeled_indices)
 
# Create DataLoader for labeled and unlabeled data
labeled_loader = DataLoader(labeled_data, batch_size=32, shuffle=True)
unlabeled_loader = DataLoader(unlabeled_data, batch_size=32, shuffle=False)
 
# Initialize the model, optimizer, and loss function
model = SimpleCNN()
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()
 
# Active learning loop
for iteration in range(10):  # Active learning loop for 10 iterations
    model.train()
    total_loss = 0
    for data, target in labeled_loader:
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
 
    print(f"Iteration {iteration+1}, Loss: {total_loss / len(labeled_loader)}")
 
    # Uncertainty sampling: query the most uncertain samples from the unlabeled set
    uncertain_indices = uncertainty_sampling(model, unlabeled_loader, n_samples=10)
 
    # Add the queried samples to the labeled set
    labeled_indices.extend(uncertain_indices)
    unlabeled_indices = list(set(range(len(dataset))) - set(labeled_indices))
 
    # Create new DataLoaders with updated labeled and unlabeled datasets
    labeled_data = torch.utils.data.Subset(dataset, labeled_indices)
    unlabeled_data = torch.utils.data.Subset(dataset, unlabeled_indices)
 
    labeled_loader = DataLoader(labeled_data, batch_size=32, shuffle=True)
    unlabeled_loader = DataLoader(unlabeled_data, batch_size=32, shuffle=False)
 
# Evaluate the model on the full CIFAR-10 test set
model.eval()
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
 
correct = 0
total = 0
with torch.no_grad():
    for data, target in test_loader:
        output = model(data)
        _, predicted = torch.max(output, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()
 
print(f"Final Accuracy: {100 * correct / total:.2f}%")