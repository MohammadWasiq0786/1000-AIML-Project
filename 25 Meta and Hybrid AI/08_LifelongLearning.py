"""
Project 968: Lifelong Learning Implementation
Description
Lifelong learning systems enable models to continuously learn from new data while retaining the knowledge learned from previous tasks. In this project, we will implement a simple lifelong learning system that can learn multiple tasks sequentially without forgetting the previous tasks.

Key Concepts Covered:
Lifelong Learning: A system that learns from a sequence of tasks while retaining knowledge from previous ones.

Elastic Weight Consolidation (EWC): A technique that helps prevent catastrophic forgetting by penalizing changes to important weights.

Task Incremental Learning: The model can learn new tasks while maintaining performance on previously learned tasks.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
 
# Define a basic neural network
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(64, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
 
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
 
# EWC loss function (Elastic Weight Consolidation)
class EWC:
    def __init__(self, model, dataloader, importance_factor=1000):
        self.model = model
        self.importance_factor = importance_factor
        self.saved_params = {}
        self.saved_fisher_information = {}
        self._capture_params(dataloader)
 
    def _capture_params(self, dataloader):
        # Save model parameters
        for name, param in self.model.named_parameters():
            self.saved_params[name] = param.clone().detach()
 
        # Compute Fisher Information
        fisher_information = {}
        for name, param in self.model.named_parameters():
            fisher_information[name] = torch.zeros_like(param)
 
        self.model.eval()
        for data, target in dataloader:
            self.model.zero_grad()
            output = self.model(data)
            loss = nn.CrossEntropyLoss()(output, target)
            loss.backward()
 
            for name, param in self.model.named_parameters():
                fisher_information[name] += param.grad ** 2 / len(dataloader)
 
        self.saved_fisher_information = fisher_information
 
    def compute_ewc_loss(self):
        ewc_loss = 0
        for name, param in self.model.named_parameters():
            fisher_information = self.saved_fisher_information[name]
            old_param = self.saved_params[name]
            ewc_loss += (fisher_information * (param - old_param) ** 2).sum()
 
        return self.importance_factor * ewc_loss
 
# Load Digits dataset (for simplicity)
digits = load_digits()
X = digits.data / 16.0  # Normalize data
y = digits.target
 
# Split into two tasks (Task 1: First 5 classes, Task 2: Last 5 classes)
X_task1, X_task2 = X[y < 5], X[y >= 5]
y_task1, y_task2 = y[y < 5], y[y >= 5]
 
# Split into training and testing sets for both tasks
X_train1, X_test1, y_train1, y_test1 = train_test_split(X_task1, y_task1, test_size=0.2, random_state=42)
X_train2, X_test2, y_train2, y_test2 = train_test_split(X_task2, y_task2, test_size=0.2, random_state=42)
 
# Convert to PyTorch tensors
train_data1 = TensorDataset(torch.tensor(X_train1, dtype=torch.float32), torch.tensor(y_train1, dtype=torch.long))
test_data1 = TensorDataset(torch.tensor(X_test1, dtype=torch.float32), torch.tensor(y_test1, dtype=torch.long))
 
train_data2 = TensorDataset(torch.tensor(X_train2, dtype=torch.float32), torch.tensor(y_train2, dtype=torch.long))
test_data2 = TensorDataset(torch.tensor(X_test2, dtype=torch.float32), torch.tensor(y_test2, dtype=torch.long))
 
# Create DataLoader
train_loader1 = DataLoader(train_data1, batch_size=32, shuffle=True)
test_loader1 = DataLoader(test_data1, batch_size=32, shuffle=False)
 
train_loader2 = DataLoader(train_data2, batch_size=32, shuffle=True)
test_loader2 = DataLoader(test_data2, batch_size=32, shuffle=False)
 
# Initialize the model, optimizer, and EWC
model = SimpleNN()
optimizer = optim.Adam(model.parameters(), lr=0.001)
 
# Task 1: Train on the first task
for epoch in range(5):
    model.train()
    total_loss = 0
    for data, target in train_loader1:
        optimizer.zero_grad()
        output = model(data)
        loss = nn.CrossEntropyLoss()(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
 
    print(f"Task 1 Epoch {epoch+1}, Loss: {total_loss / len(train_loader1)}")
 
# Capture parameters and Fisher information after Task 1
ewc = EWC(model, train_loader1)
 
# Task 2: Train on the second task using EWC to avoid forgetting Task 1
for epoch in range(5):
    model.train()
    total_loss = 0
    for data, target in train_loader2:
        optimizer.zero_grad()
        output = model(data)
        loss = nn.CrossEntropyLoss()(output, target)
 
        # Add the EWC loss term
        ewc_loss = ewc.compute_ewc_loss()
        total_loss = loss + ewc_loss
        total_loss.backward()
        optimizer.step()
 
    print(f"Task 2 Epoch {epoch+1}, Loss: {total_loss / len(train_loader2)}")
 
# Evaluate on Task 1 and Task 2
model.eval()
correct1, correct2 = 0, 0
total1, total2 = 0, 0
with torch.no_grad():
    for data, target in test_loader1:
        output = model(data)
        _, predicted = torch.max(output, 1)
        total1 += target.size(0)
        correct1 += (predicted == target).sum().item()
 
    for data, target in test_loader2:
        output = model(data)
        _, predicted = torch.max(output, 1)
        total2 += target.size(0)
        correct2 += (predicted == target).sum().item()
 
print(f"Task 1 Accuracy: {100 * correct1 / total1:.2f}%")
print(f"Task 2 Accuracy: {100 * correct2 / total2:.2f}%")