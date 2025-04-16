"""
Project 157. Federated learning simulation
Description:
Federated Learning enables training machine learning models across multiple decentralized devices (clients) without sharing their raw data. Instead of sending data to a central server, clients train locally and share only model updates. This project simulates a federated learning setup with multiple clients on the MNIST dataset.

About:
🧠 What This Project Demonstrates:
Simulates federated learning rounds across multiple virtual clients

Performs local model training and federated averaging (FedAvg)

Trains a global model without centralizing user data
"""

# Install if not already: pip install torch torchvision
 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import copy
 
# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 
# Simple CNN model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 128), nn.ReLU(),
            nn.Linear(128, 10)
        )
    def forward(self, x):
        return self.net(x)
 
# Load and partition MNIST into multiple clients
transform = transforms.ToTensor()
full_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
num_clients = 5
client_data_size = len(full_dataset) // num_clients
client_datasets = [Subset(full_dataset, list(range(i * client_data_size, (i + 1) * client_data_size)))
                   for i in range(num_clients)]
client_loaders = [DataLoader(ds, batch_size=32, shuffle=True) for ds in client_datasets]
 
# Local training for one client
def train_local(model, data_loader, epochs=1):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()
    for _ in range(epochs):
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return model.state_dict()
 
# Federated averaging
def federated_average(weights_list):
    avg_weights = copy.deepcopy(weights_list[0])
    for key in avg_weights:
        for i in range(1, len(weights_list)):
            avg_weights[key] += weights_list[i][key]
        avg_weights[key] = avg_weights[key] / len(weights_list)
    return avg_weights
 
# Central test set
test_loader = DataLoader(datasets.MNIST('./data', train=False, transform=transform),
                         batch_size=1000, shuffle=False)
 
# Evaluation function
def evaluate(model, loader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            preds = model(images).argmax(dim=1)
            correct += (preds == labels).sum().item()
    return correct / len(loader.dataset)
 
# Federated training loop
global_model = CNN().to(device)
print("🌐 Starting Federated Learning...")
 
for round in range(3):  # Simulate 3 global rounds
    local_weights = []
    for client_id, loader in enumerate(client_loaders):
        local_model = copy.deepcopy(global_model)
        local_weights.append(train_local(local_model.to(device), loader))
        print(f"🖥️ Client {client_id+1} done training.")
 
    # Aggregate weights
    avg_weights = federated_average(local_weights)
    global_model.load_state_dict(avg_weights)
 
    # Evaluate
    acc = evaluate(global_model, test_loader)
    print(f"✅ Round {round+1} - Global Model Accuracy: {acc:.2%}")