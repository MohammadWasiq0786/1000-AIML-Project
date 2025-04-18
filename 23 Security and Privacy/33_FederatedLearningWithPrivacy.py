"""
Project 913. Federated Learning with Privacy

Federated learning (FL) allows multiple devices or parties to collaboratively train a model without sharing their raw data. In this project, we simulate federated learning across multiple clients and integrate differential privacy during updates.

What This Demonstrates:
Data never leaves the client

Updates are locally trained and then aggregated

Noise is added to gradients to simulate privacy-preserving updates (DP-FL)

📦 In real-world FL systems:

Use libraries like Flower, PySyft, TensorFlow Federated

Add secure aggregation so even the server can’t see individual updates

Deploy on devices for personalized models (e.g., Gboard, Apple devices)
"""

import torch
from torch import nn, optim
import copy
 
# Simulate 3 clients with small, private datasets
client_data = [
    (torch.randn(20, 5), torch.randint(0, 2, (20,)).float()),
    (torch.randn(20, 5), torch.randint(0, 2, (20,)).float()),
    (torch.randn(20, 5), torch.randint(0, 2, (20,)).float()),
]
 
# Shared model architecture (binary classifier)
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(5, 1)
    
    def forward(self, x):
        return torch.sigmoid(self.fc(x))
 
# Training on client with differential privacy (simulated by adding noise to gradients)
def train_local(model, data, labels, epochs=1, lr=0.1, noise_std=0.1):
    model = copy.deepcopy(model)
    optimizer = optim.SGD(model.parameters(), lr=lr)
    loss_fn = nn.BCELoss()
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        outputs = model(data).squeeze()
        loss = loss_fn(outputs, labels)
        loss.backward()
        
        # Add Gaussian noise to gradients for DP
        for param in model.parameters():
            param.grad += torch.randn_like(param.grad) * noise_std
        
        optimizer.step()
    return model.state_dict()
 
# Initialize global model
global_model = SimpleModel()
global_state = global_model.state_dict()
 
# Perform one round of federated learning with DP updates
client_states = []
for data, labels in client_data:
    local_model = SimpleModel()
    local_model.load_state_dict(global_state)
    updated_state = train_local(local_model, data, labels)
    client_states.append(updated_state)
 
# Aggregate updates (simple average)
new_global_state = copy.deepcopy(global_state)
for key in global_state:
    new_global_state[key] = sum(client[key] for client in client_states) / len(client_states)
 
# Update global model
global_model.load_state_dict(new_global_state)
print("✅ Federated learning round complete with privacy-preserving updates.")