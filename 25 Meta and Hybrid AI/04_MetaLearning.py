"""
Project 964: Meta-learning Implementation
Description
Meta-learning, or "learning to learn," enables models to adapt to new tasks with minimal data. It focuses on training models to generalize across tasks. In this project, we’ll implement a meta-learning system using model-agnostic meta-learning (MAML), which allows a model to learn quickly from a small number of examples.

Key Concepts Covered:
Model-Agnostic Meta-Learning (MAML): A meta-learning algorithm that trains models to quickly adapt to new tasks.

Few-shot learning: Training a model to perform well with only a few examples from a new task.

Meta-training Loop: The process of training the model on multiple tasks, optimizing it for fast adaptation.
"""

# pip install learn2learn

import learn2learn as l2l
import torch
from torch import nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
 
# Define a simple neural network for classification
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.fc1 = nn.Linear(64 * 6 * 6, 128)
        self.fc2 = nn.Linear(128, 10)  # 10 output classes for MNIST
 
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
 
# Load MNIST dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
mnist_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
 
# Split dataset into training and validation sets
train_dataset, val_dataset = random_split(mnist_dataset, [55000, 5000])
 
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
 
# Create a meta-learning model using MAML
model = SimpleCNN()
maml = l2l.algorithms.MAML(model, lr=0.01)
 
# Meta-training loop (training on multiple tasks)
optimizer = optim.Adam(maml.parameters(), lr=0.001)
 
for epoch in range(5):  # Meta-training for 5 epochs
    total_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        # Meta-learning step
        loss = maml(data, target)  # Calculate loss for the task
        total_loss += loss.item()
 
        # Step the optimizer to update model parameters
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
 
    print(f'Epoch {epoch+1}, Loss: {total_loss/len(train_loader)}')
 
# Test the meta-learned model on validation data
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for data, target in val_loader:
        output = model(data)
        _, predicted = torch.max(output, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()
 
print(f'Accuracy on validation data: {100 * correct / total:.2f}%')