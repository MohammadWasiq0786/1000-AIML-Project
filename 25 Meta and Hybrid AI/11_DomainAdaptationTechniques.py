"""
Project 971: Domain Adaptation Techniques
Description
Domain adaptation involves transferring knowledge learned from a source domain (with plenty of labeled data) to a target domain (with limited or no labeled data). In this project, we’ll implement a domain adaptation technique, such as adversarial training or fine-tuning, to adapt a model trained on one dataset to perform well on another.

Key Concepts Covered:
Adversarial Training: In domain adaptation, adversarial training helps the model learn to classify from the target domain by "fooling" a domain classifier into thinking that the source domain examples are from the target domain.

Domain Adaptation: A technique that enables the model to transfer knowledge from a source domain to a target domain, even if labeled data is scarce for the target domain.

Domain Classifier: A separate network that learns to distinguish between source and target domain examples, guiding the main model to learn generalizable features.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
 
# Define the model architecture
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
        self.fc1 = nn.Linear(1024, 128)
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
 
# Define the domain classifier (for adversarial training)
class DomainClassifier(nn.Module):
    def __init__(self):
        super(DomainClassifier, self).__init__()
        self.fc1 = nn.Linear(1024, 128)
        self.fc2 = nn.Linear(128, 1)
 
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
 
# Define the adversarial loss function (Domain Adaptation Loss)
def adversarial_loss(domain_output, target_domain):
    return nn.BCEWithLogitsLoss()(domain_output, target_domain)
 
# Load the MNIST and USPS datasets (Source and Target Domains)
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
 
mnist_train = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
usps_train = datasets.USPS(root='./data', train=True, download=True, transform=transform)
 
# Create DataLoaders
mnist_loader = DataLoader(mnist_train, batch_size=64, shuffle=True)
usps_loader = DataLoader(usps_train, batch_size=64, shuffle=True)
 
# Initialize models
cnn_model = SimpleCNN()
domain_classifier = DomainClassifier()
 
# Optimizers
optimizer_cnn = optim.Adam(cnn_model.parameters(), lr=0.001)
optimizer_domain = optim.Adam(domain_classifier.parameters(), lr=0.001)
 
# Train the model with adversarial domain adaptation
for epoch in range(5):
    cnn_model.train()
    domain_classifier.train()
 
    total_loss = 0
    for (source_data, _), (target_data, _) in zip(mnist_loader, usps_loader):
        # Adversarial training
        source_domain_label = torch.ones(source_data.size(0), 1)  # Source domain label: 1
        target_domain_label = torch.zeros(target_data.size(0), 1)  # Target domain label: 0
 
        # Forward pass through CNN (shared layers)
        source_features = cnn_model(source_data)
        target_features = cnn_model(target_data)
 
        # Adversarial loss: try to confuse the domain classifier
        source_domain_output = domain_classifier(source_features)
        target_domain_output = domain_classifier(target_features)
 
        domain_loss = adversarial_loss(source_domain_output, source_domain_label) + adversarial_loss(target_domain_output, target_domain_label)
 
        # Backpropagation for domain classifier
        optimizer_domain.zero_grad()
        domain_loss.backward()
        optimizer_domain.step()
 
        # Train CNN to minimize domain loss (domain classifier confusion)
        optimizer_cnn.zero_grad()
        total_loss += domain_loss.item()
        domain_loss.backward()
        optimizer_cnn.step()
 
    print(f"Epoch {epoch+1}, Domain Loss: {total_loss / len(mnist_loader)}")
 
# Test the model on the target domain (USPS)
cnn_model.eval()
correct = 0
total = 0
with torch.no_grad():
    for data, target in usps_loader:
        output = cnn_model(data)
        _, predicted = torch.max(output, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()
 
print(f"Accuracy on target domain (USPS): {100 * correct / total:.2f}%")