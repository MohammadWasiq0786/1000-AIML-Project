"""
Project 141. Neural architecture search
Description:
Neural Architecture Search (NAS) is an AutoML technique that automatically finds the best-performing neural network architecture for a given task. Instead of manually tuning layers and connections, NAS explores the architecture space using search strategies like random search, reinforcement learning, or evolutionary algorithms. In this project, we implement a basic random search NAS to find the best architecture for image classification on MNIST.

About:
🧠 What This Project Demonstrates:
Implements basic NAS via random search

Automatically tests different fully connected architectures

Trains on MNIST and picks the best-performing model
"""

# Install if not already: pip install torch torchvision matplotlib
 
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import random
 
# Load MNIST
transform = transforms.Compose([transforms.ToTensor()])
train_set = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
val_set = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
 
train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=1000, shuffle=False)
 
# Define random architecture generator
def generate_model(input_size=784, output_size=10):
    layers = []
    current_size = input_size
    num_layers = random.randint(1, 3)
    for _ in range(num_layers):
        next_size = random.choice([64, 128, 256])
        layers.append(nn.Linear(current_size, next_size))
        layers.append(nn.ReLU())
        current_size = next_size
    layers.append(nn.Linear(current_size, output_size))
    return nn.Sequential(*layers)
 
# Train + evaluate a model
def train_and_evaluate(model, epochs=2):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
 
    # Training loop
    model.train()
    for epoch in range(epochs):
        for images, labels in train_loader:
            images = images.view(images.size(0), -1).to(device)
            labels = labels.to(device)
 
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
 
    # Evaluation
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.view(images.size(0), -1).to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total  # Accuracy
 
# Run NAS: Random search over architectures
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
best_acc = 0
best_model = None
 
print("🔍 Starting Neural Architecture Search...")
for i in range(5):  # Try 5 different architectures
    model = generate_model()
    acc = train_and_evaluate(model)
    print(f"Model {i+1}: Accuracy = {acc:.4f}")
    if acc > best_acc:
        best_acc = acc
        best_model = model
 
print(f"\n🏆 Best Model Accuracy: {best_acc:.4f}")