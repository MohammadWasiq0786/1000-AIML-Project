"""
Project 980: Model Compression Techniques
Description
Model compression refers to the process of reducing the size of a neural network while maintaining its performance. Techniques like pruning, quantization, and weight sharing can be used to make models smaller, faster, and more efficient, which is particularly useful for deployment on edge devices or mobile applications. In this project, we will implement pruning and quantization techniques for compressing a neural network.

Key Concepts Covered:
Pruning: A technique for removing unimportant weights from the network (i.e., setting them to zero) to reduce the model's size and complexity. Here, we used random pruning, but more advanced methods can be explored.

Quantization: Reducing the precision of the weights and activations (e.g., from 32-bit floating-point to 8-bit integers) to decrease model size and improve inference speed on hardware accelerators.

Post-training Quantization: Quantizing a pre-trained model, which is useful when you have a trained model and want to optimize it for deployment without retraining.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import torch.nn.utils.prune as prune
 
# Define a simple CNN for model compression
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.fc1 = nn.Linear(64 * 6 * 6, 128)
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
 
# Load CIFAR-10 dataset
transform = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()])
train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
 
# Initialize the model, optimizer, and loss function
model = SimpleCNN()
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()
 
# Training loop before compression (Baseline model)
for epoch in range(2):  # Training for 2 epochs (Baseline)
    model.train()
    total_loss = 0
    for data, target in train_loader:
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
 
    print(f"Epoch {epoch+1}, Loss: {total_loss / len(train_loader)}")
 
# 1. Model Pruning - Prune 20% of the weights in each layer
prune.random_unstructured(model.conv1, name="weight", amount=0.2)
prune.random_unstructured(model.conv2, name="weight", amount=0.2)
prune.random_unstructured(model.fc1, name="weight", amount=0.2)
 
# Check how many parameters are pruned
print("\nPruned model parameters:")
print(f"Conv1 pruned: {prune.get_pruned_parameters(model.conv1)}")
print(f"Conv2 pruned: {prune.get_pruned_parameters(model.conv2)}")
print(f"Fc1 pruned: {prune.get_pruned_parameters(model.fc1)}")
 
# 2. Apply Quantization (Post-training static quantization)
model.eval()  # Set the model to evaluation mode
 
# Fuse the Conv + ReLU layers (for better quantization results)
model = torch.quantization.fuse_modules(model, [['conv1', 'relu']])
model = torch.quantization.fuse_modules(model, [['conv2', 'relu']])
 
# Prepare the model for quantization
model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
torch.quantization.prepare(model, inplace=True)
 
# Calibrate with a few batches to determine the scale and zero point
with torch.no_grad():
    for data, _ in train_loader:
        model(data)
 
# Convert to a quantized model
torch.quantization.convert(model, inplace=True)
 
# Evaluate the pruned and quantized model
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for data, target in train_loader:
        output = model(data)
        _, predicted = torch.max(output, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()
 
print(f"\nAccuracy of the pruned and quantized model: {100 * correct / total:.2f}%")