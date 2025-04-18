"""
Project 965: Learning to Learn Algorithms
Description
Learning to learn algorithms, also known as meta-learning algorithms, aim to optimize a model’s ability to generalize and adapt quickly to new tasks with minimal data. This project focuses on implementing a simple meta-learning algorithm, such as MAML or Reptile, for few-shot learning tasks like classification or regression.

Key Concepts Covered:
Reptile Meta-Learning: A simpler meta-learning algorithm that focuses on updating the model parameters to quickly adapt to new tasks.

Few-Shot Learning: Training models to perform well with only a few examples from a new task.

Meta-Training Loop: The outer loop that trains the model on multiple tasks, ensuring it can quickly adapt to new ones.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import learn2learn as l2l
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
 
# Define a simple neural network for few-shot classification
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(28*28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
 
    def forward(self, x):
        x = x.view(-1, 28*28)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
 
# Define the Reptile meta-learning algorithm
def reptile_update(model, meta_lr=0.1):
    for param in model.parameters():
        param.data -= meta_lr * param.grad.data
 
# Load Omniglot dataset (used for few-shot learning)
transform = transforms.Compose([transforms.Grayscale(num_output_channels=1), transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
omniglot_train = datasets.Omniglot(root='./data', background=True, transform=transform, download=True)
omniglot_test = datasets.Omniglot(root='./data', background=False, transform=transform, download=True)
 
# Create DataLoader for training and testing
train_loader = DataLoader(omniglot_train, batch_size=32, shuffle=True)
test_loader = DataLoader(omniglot_test, batch_size=32, shuffle=False)
 
# Initialize the model
model = SimpleNN()
optimizer = optim.Adam(model.parameters(), lr=0.001)
 
# Train the model using Reptile meta-learning algorithm
for epoch in range(5):  # Meta-training loop for 5 epochs
    model.train()
    total_loss = 0
 
    # Task-specific training loop
    for data, target in train_loader:
        optimizer.zero_grad()
        
        # Task training: forward pass and loss calculation
        output = model(data)
        loss = nn.CrossEntropyLoss()(output, target)
        loss.backward()
        
        # Update the model with the Reptile algorithm
        reptile_update(model, meta_lr=0.1)
        
        total_loss += loss.item()
 
    print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader)}")
 
# Test the model on the test dataset
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for data, target in test_loader:
        output = model(data)
        _, predicted = torch.max(output, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()
 
print(f"Accuracy on test data: {100 * correct / total:.2f}%")