"""
Project 152. Multi-task learning system
Description:
Multi-Task Learning (MTL) is a technique where a model learns to perform multiple tasks simultaneously by sharing representations. It improves generalization and reduces training cost by jointly optimizing for related objectives. In this project, we implement an MTL system that performs digit classification and even/odd prediction on the MNIST dataset using a shared CNN encoder.

About:
🧠 What This Project Demonstrates:
Builds a shared CNN for multiple tasks

Learns both digit classification (10-class) and even/odd (binary)

Trains with joint loss to optimize both heads
"""

# Install if not already: pip install torch torchvision matplotlib
 
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
 
# Dataset loader with dual labels: class (0–9) and even/odd
class MultiTaskMNIST(torch.utils.data.Dataset):
    def __init__(self, train=True):
        self.mnist = datasets.MNIST(root='./data', train=train, download=True, transform=transforms.ToTensor())
 
    def __len__(self):
        return len(self.mnist)
 
    def __getitem__(self, idx):
        image, label = self.mnist[idx]
        even_odd = label % 2  # 0 = even, 1 = odd
        return image, label, even_odd
 
train_loader = DataLoader(MultiTaskMNIST(train=True), batch_size=64, shuffle=True)
test_loader = DataLoader(MultiTaskMNIST(train=False), batch_size=1000)
 
# Multi-task CNN model
class MultiTaskCNN(nn.Module):
    def __init__(self):
        super(MultiTaskCNN, self).__init__()
        self.shared = nn.Sequential(
            nn.Conv2d(1, 32, 3), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten()
        )
        self.class_head = nn.Linear(64 * 5 * 5, 10)  # Digit classification
        self.binary_head = nn.Linear(64 * 5 * 5, 2)  # Even/Odd classification
 
    def forward(self, x):
        features = self.shared(x)
        digit_out = self.class_head(features)
        binary_out = self.binary_head(features)
        return digit_out, binary_out
 
# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MultiTaskCNN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn_digit = nn.CrossEntropyLoss()
loss_fn_evenodd = nn.CrossEntropyLoss()
 
# Training loop (1 epoch demo)
model.train()
for images, labels, even_odd in train_loader:
    images, labels, even_odd = images.to(device), labels.to(device), even_odd.to(device)
 
    digit_logits, binary_logits = model(images)
 
    loss_digit = loss_fn_digit(digit_logits, labels)
    loss_binary = loss_fn_evenodd(binary_logits, even_odd)
    loss = loss_digit + loss_binary  # Combine both tasks
 
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
 
print("✅ One training epoch complete.")
 
# Evaluation (simple accuracy check)
model.eval()
with torch.no_grad():
    correct_digits = 0
    correct_evenodd = 0
    total = 0
    for images, labels, even_odd in test_loader:
        images, labels, even_odd = images.to(device), labels.to(device), even_odd.to(device)
        digit_logits, binary_logits = model(images)
        _, pred_digits = digit_logits.max(1)
        _, pred_evenodd = binary_logits.max(1)
        correct_digits += (pred_digits == labels).sum().item()
        correct_evenodd += (pred_evenodd == even_odd).sum().item()
        total += labels.size(0)
 
print(f"🧠 Digit Classification Accuracy: {correct_digits / total:.2%}")
print(f"🧠 Even/Odd Accuracy: {correct_evenodd / total:.2%}")