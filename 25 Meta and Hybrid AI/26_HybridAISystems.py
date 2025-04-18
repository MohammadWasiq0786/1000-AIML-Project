"""
Project 986: Hybrid AI Systems
Description
Hybrid AI systems combine multiple AI approaches, such as symbolic reasoning and machine learning, to leverage the strengths of each. This hybrid approach allows the system to handle both complex, data-driven tasks and tasks requiring explicit reasoning or logic. In this project, we will implement a hybrid AI system that combines machine learning with rule-based reasoning to solve a simple classification task.

Key Concepts Covered:
Hybrid AI Systems: Combining machine learning (data-driven) and symbolic reasoning (logic and rules) to solve tasks more effectively.

Neural Networks for Learning: The neural network learns data representations and makes predictions based on the data.

Symbolic Reasoning: Symbolic rules (e.g., threshold-based decision-making) refine or adjust the predictions from the neural network, enhancing interpretability and decision-making.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import make_classification
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
 
# Define a simple neural network model (machine learning component)
class SimpleMLModel(nn.Module):
    def __init__(self):
        super(SimpleMLModel, self).__init__()
        self.fc1 = nn.Linear(2, 64)  # First layer with 2 input features
        self.fc2 = nn.Linear(64, 1)  # Output layer (binary classification)
 
    def forward(self, x):
        x = torch.relu(self.fc1(x))  # Activation function for non-linearity
        x = torch.sigmoid(self.fc2(x))  # Output layer with sigmoid for binary classification
        return x
 
# Generate synthetic classification data (binary)
X, y = make_classification(n_samples=1000, n_features=2, n_informative=2, n_redundant=0, random_state=42)
 
# Convert data to PyTorch tensors
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)
 
# Create DataLoader for training
dataset = TensorDataset(X_tensor, y_tensor)
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
 
# Initialize the machine learning model, optimizer, and loss function
ml_model = SimpleMLModel()
optimizer = optim.Adam(ml_model.parameters(), lr=0.001)
criterion = nn.BCELoss()  # Binary Cross Entropy Loss for binary classification
 
# Train the machine learning model
for epoch in range(5):  # Training for 5 epochs
    ml_model.train()
    total_loss = 0
    for data, target in train_loader:
        optimizer.zero_grad()
        output = ml_model(data)  # Make predictions
        loss = criterion(output, target)  # Calculate loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
 
    print(f"Epoch {epoch+1}, Loss: {total_loss / len(train_loader)}")
 
# Define symbolic rules for decision refinement (symbolic AI component)
def symbolic_rules(predictions):
    """
    Apply symbolic rules to adjust or refine model predictions.
    For example, if the prediction is greater than 0.8, classify as 1 (True).
    """
    adjusted_predictions = torch.where(predictions > 0.8, torch.ones_like(predictions), predictions)
    return adjusted_predictions
 
# Evaluate the model with symbolic rules
ml_model.eval()
with torch.no_grad():
    test_data = torch.tensor([[0.3, 0.4], [0.8, 0.9], [0.2, 0.7]], dtype=torch.float32)
    raw_predictions = ml_model(test_data)  # Get raw predictions from the model
    print(f"Raw Predictions: {raw_predictions.squeeze().numpy()}")
 
    # Apply symbolic rules to adjust the predictions
    refined_predictions = symbolic_rules(raw_predictions)
    print(f"Refined Predictions after Symbolic Rules: {refined_predictions.squeeze().numpy()}")
 
# Plot the decision boundary before and after symbolic reasoning
plt.figure(figsize=(10, 6))
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', s=25, edgecolors='k', alpha=0.6)
plt.title("Data Points with Hybrid AI Decision Boundaries")
 
# Plot the decision boundary of the machine learning model (before symbolic rules)
xx, yy = np.meshgrid(np.linspace(X[:, 0].min(), X[:, 0].max(), 100), np.linspace(X[:, 1].min(), X[:, 1].max(), 100))
grid = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32)
predictions = ml_model(grid).detach().numpy().reshape(xx.shape)
plt.contourf(xx, yy, predictions, levels=np.linspace(0, 1, 11), cmap='coolwarm', alpha=0.3)
 
plt.show()