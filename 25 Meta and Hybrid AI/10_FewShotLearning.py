"""
Project 970: Few-shot Learning Implementation
Description
Few-shot learning allows a model to generalize and make predictions with just a few labeled examples. In this project, we will implement a simple few-shot learning system using Siamese networks, which are designed for tasks like one-shot and few-shot image classification.

Key Concepts Covered:
Siamese Networks: A network architecture that compares two inputs and determines whether they belong to the same class (ideal for few-shot learning).

Few-shot Learning: A machine learning paradigm where a model is trained to generalize from a small number of labeled examples.

Pairwise Comparison: The core idea in Siamese Networks, where the model compares pairs of inputs and learns whether they belong to the same class.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np
 
# Define the Siamese Network architecture
class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3)
        self.fc1 = nn.Linear(128*6*6, 256)
        self.fc2 = nn.Linear(256, 1)
 
    def forward_one(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.flatten(x, start_dim=1)
        x = torch.relu(self.fc1(x))
        return x
 
    def forward(self, input1, input2):
        out1 = self.forward_one(input1)
        out2 = self.forward_one(input2)
        diff = torch.abs(out1 - out2)  # Calculate the absolute difference between the outputs
        return torch.sigmoid(self.fc2(diff))  # Output similarity score (0 or 1)
 
# Define the dataset class for Omniglot (few-shot task setup)
class OmniglotFewShot(Dataset):
    def __init__(self, data, labels, num_classes=5, num_examples=5):
        self.data = data
        self.labels = labels
        self.num_classes = num_classes
        self.num_examples = num_examples
 
    def __getitem__(self, idx):
        # Select two random classes
        class1_idx = np.random.randint(0, self.num_classes)
        class2_idx = np.random.randint(0, self.num_classes)
 
        # Get example pairs from each class
        class1_data = self.data[self.labels == class1_idx]
        class2_data = self.data[self.labels == class2_idx]
 
        # Select random examples for both classes
        example1 = class1_data[np.random.randint(len(class1_data))]
        example2 = class2_data[np.random.randint(len(class2_data))]
 
        # Label the pair as either a match (1) or non-match (0)
        label = 1 if class1_idx == class2_idx else 0
 
        return example1, example2, label
 
    def __len__(self):
        return len(self.data)
 
# Load Omniglot dataset (for simplicity, use a subset)
transform = transforms.Compose([transforms.Grayscale(num_output_channels=1), transforms.ToTensor()])
omniglot_dataset = datasets.Omniglot(root='./data', background=True, transform=transform, download=True)
 
# Convert to tensor format
data = omniglot_dataset.data.numpy().reshape(-1, 1, 28, 28) / 255.0
labels = omniglot_dataset.targets.numpy()
 
# Set up the few-shot dataset
few_shot_dataset = OmniglotFewShot(data, labels)
 
# DataLoader for few-shot learning
train_loader = DataLoader(few_shot_dataset, batch_size=32, shuffle=True)
 
# Initialize the model, optimizer, and loss function
model = SiameseNetwork()
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.BCELoss()
 
# Train the model
for epoch in range(5):  # Few-shot training loop for 5 epochs
    model.train()
    total_loss = 0
    for data1, data2, label in train_loader:
        optimizer.zero_grad()
        output = model(data1, data2)
        loss = criterion(output.squeeze(), label.float())
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
 
    print(f"Epoch {epoch+1}, Loss: {total_loss / len(train_loader)}")
 
# Test the model
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for data1, data2, label in train_loader:
        output = model(data1, data2)
        _, predicted = torch.max(output, 1)
        total += label.size(0)
        correct += (predicted == label).sum().item()
 
print(f"Accuracy on few-shot task: {100 * correct / total:.2f}%")