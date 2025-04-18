"""
Project 967: Progressive Neural Networks
Description
Progressive neural networks allow a model to adapt to new tasks by adding new neural network components while retaining previously learned knowledge. This project focuses on implementing progressive neural networks to learn a sequence of tasks while preserving prior task knowledge.

Key Concepts Covered:
Progressive Neural Networks: A method for continual learning where new "columns" are added for new tasks while old tasks are retained by freezing the previous columns.

Task Incremental Learning: Adapting the model to new tasks without forgetting previous ones.

Frozen Parameters: Freezing the parameters of previous layers to preserve knowledge when learning a new task.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
 
# Define a basic neural network with two columns for progressive learning
class ProgressiveNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ProgressiveNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.fc3 = nn.Linear(hidden_size, output_size)  # New column for the second task
        self.fc1.weight.data.normal_()  # Initialize weights
        self.fc2.weight.data.normal_()
        self.fc3.weight.data.normal_()
 
    def forward(self, x, task=1):
        x = torch.relu(self.fc1(x))
        if task == 1:
            x = self.fc2(x)  # Use first column for Task 1
        elif task == 2:
            x = self.fc3(x)  # Use second column for Task 2
        return x
 
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
 
# Create DataLoader for training and testing
train_loader1 = DataLoader(train_data1, batch_size=32, shuffle=True)
test_loader1 = DataLoader(test_data1, batch_size=32, shuffle=False)
 
train_loader2 = DataLoader(train_data2, batch_size=32, shuffle=True)
test_loader2 = DataLoader(test_data2, batch_size=32, shuffle=False)
 
# Initialize the model and optimizer
model = ProgressiveNN(input_size=64, hidden_size=128, output_size=5)
optimizer = optim.Adam(model.parameters(), lr=0.001)
 
# Task 1: Train on the first task
for epoch in range(5):
    model.train()
    total_loss = 0
    for data, target in train_loader1:
        optimizer.zero_grad()
        output = model(data, task=1)
        loss = nn.CrossEntropyLoss()(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
 
    print(f"Task 1 Epoch {epoch+1}, Loss: {total_loss / len(train_loader1)}")
 
# Task 2: Train on the second task using the second column (frozen first column)
for epoch in range(5):
    model.train()
    total_loss = 0
    for data, target in train_loader2:
        optimizer.zero_grad()
        output = model(data, task=2)
        loss = nn.CrossEntropyLoss()(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
 
    print(f"Task 2 Epoch {epoch+1}, Loss: {total_loss / len(train_loader2)}")
 
# Evaluate on Task 1 and Task 2
model.eval()
correct1, correct2 = 0, 0
total1, total2 = 0, 0
with torch.no_grad():
    for data, target in test_loader1:
        output = model(data, task=1)
        _, predicted = torch.max(output, 1)
        total1 += target.size(0)
        correct1 += (predicted == target).sum().item()
 
    for data, target in test_loader2:
        output = model(data, task=2)
        _, predicted = torch.max(output, 1)
        total2 += target.size(0)
        correct2 += (predicted == target).sum().item()
 
print(f"Task 1 Accuracy: {100 * correct1 / total1:.2f}%")
print(f"Task 2 Accuracy: {100 * correct2 / total2:.2f}%")