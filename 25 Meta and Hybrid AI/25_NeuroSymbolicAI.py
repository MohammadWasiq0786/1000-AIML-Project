"""
Project 985: Neuro-symbolic AI Implementation
Description
Neuro-symbolic AI combines the strengths of neural networks (learning from data) and symbolic reasoning (logic and rules). This hybrid approach allows AI systems to reason about the world while also learning from raw data. In this project, we will implement a simple neuro-symbolic system that uses both neural networks and symbolic logic to solve a task.

Key Concepts Covered:
Neuro-symbolic AI: Combines neural networks for learning and symbolic reasoning (logical rules) for decision-making.

Neural Networks for Feature Learning: The network learns useful features from data.

Symbolic Reasoning: The system applies logical rules (e.g., "if condition then action") to the predictions made by the neural network.

Combining Learning and Logic: This allows for combining data-driven learning with human-readable symbolic reasoning.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
 
# Define a simple neural network for feature learning
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(2, 64)
        self.fc2 = nn.Linear(64, 1)
 
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
 
# Define the symbolic reasoning function
def symbolic_reasoning(predictions):
    """
    Apply symbolic reasoning to the neural network predictions.
    For simplicity, assume a symbolic rule: if the prediction is > 0.5, the class is 1 (True),
    else it is 0 (False).
    """
    return predictions > 0.5
 
# Generate synthetic dataset for learning and symbolic reasoning
np.random.seed(0)
X = np.random.rand(100, 2)  # 100 samples with 2 features
y = (X[:, 0] + X[:, 1] > 1).astype(int)  # Symbolic rule: class 1 if x1 + x2 > 1, else class 0
 
# Convert to PyTorch tensors
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)
 
# Create DataLoader for training
train_data = TensorDataset(X_tensor, y_tensor)
train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
 
# Initialize the neural network model, optimizer, and loss function
model = SimpleNN()
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.BCEWithLogitsLoss()  # Binary Cross Entropy Loss
 
# Train the neural network on the data
for epoch in range(100):  # Training for 100 epochs
    model.train()
    total_loss = 0
    for data, target in train_loader:
        optimizer.zero_grad()
        output = model(data).squeeze()  # Get model predictions
        loss = criterion(output, target.squeeze())  # Calculate loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
 
    if epoch % 10 == 0:
        print(f"Epoch {epoch+1}, Loss: {total_loss / len(train_loader)}")
 
# Test the model and apply symbolic reasoning
model.eval()
with torch.no_grad():
    test_data = torch.tensor([[0.7, 0.3], [0.2, 0.9], [0.5, 0.5]], dtype=torch.float32)  # Some test inputs
    predictions = model(test_data)
    print("Predictions:", predictions)
 
    # Apply symbolic reasoning
    reasoned_results = symbolic_reasoning(predictions)
    print("Symbolic Reasoning Results (1 for True, 0 for False):", reasoned_results)