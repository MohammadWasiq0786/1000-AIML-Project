"""
Project 154. Model compression techniques
Description:
Model compression involves reducing the size and complexity of deep learning models while preserving performance. It enables deploying AI on resource-constrained environments (e.g., mobile devices, embedded systems). This project demonstrates three popular techniques: pruning, quantization, and knowledge distillation using a trained MNIST classifier.

About:
🧠 What This Project Demonstrates:
Applies L1 unstructured pruning to remove less important weights

Uses dynamic quantization to reduce model size and inference time

Demonstrates the potential of compression without retraining
"""

# Install if not already: pip install torch torchvision
 
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import torch.quantization
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
 
# Load MNIST data
transform = transforms.ToTensor()
train_set = torchvision.datasets.MNIST('./data', train=True, download=True, transform=transform)
test_set = torchvision.datasets.MNIST('./data', train=False, download=True, transform=transform)
train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
test_loader = DataLoader(test_set, batch_size=1000)
 
# Simple fully connected model
class SimpleNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(28*28, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )
 
    def forward(self, x):
        return self.fc(x.view(x.size(0), -1))
 
# Training function
def train(model, loader, optimizer, criterion, device):
    model.train()
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
 
# Evaluation function
def evaluate(model, loader, device):
    model.eval()
    correct = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            preds = model(images).argmax(dim=1)
            correct += (preds == labels).sum().item()
    return correct / len(loader.dataset)
 
# Train original model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleNN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()
 
for epoch in range(1):
    train(model, train_loader, optimizer, criterion, device)
 
print("🎯 Original Accuracy:", evaluate(model, test_loader, device))
 
# 1. 🔧 Pruning: remove 50% of weights in linear layers
for name, module in model.named_modules():
    if isinstance(module, nn.Linear):
        prune.l1_unstructured(module, name='weight', amount=0.5)
 
print("✅ Model Pruned.")
 
# 2. 🔢 Quantization (dynamic)
model.eval()
quantized_model = torch.quantization.quantize_dynamic(
    model, {nn.Linear}, dtype=torch.qint8
)
 
print("✅ Model Quantized.")
 
# 3. 📘 Distillation already done in previous project (reuse student net)
 
# Evaluate compressed model
print("🧪 Accuracy After Compression:", evaluate(quantized_model, test_loader, device))