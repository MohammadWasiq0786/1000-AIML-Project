"""
Project 156. Model pruning implementation
Description:
Model pruning reduces a neural network’s size by removing less important weights or neurons—improving inference speed and reducing memory usage with minimal accuracy loss. This project demonstrates unstructured weight pruning using PyTorch on a simple feedforward network trained on the MNIST dataset.

About:
🧠 What This Project Demonstrates:
Applies L1 unstructured pruning to dense layers

Trains a model and prunes 50% of its weights without retraining

Shows accuracy impact before and after pruning
"""

# Install dependencies:
# pip install torch torchvision
 
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
 
# Load MNIST dataset
transform = transforms.ToTensor()
train_data = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_data = datasets.MNIST('./data', train=False, download=True, transform=transform)
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=1000)
 
# Define a simple feedforward model
class SimpleMLP(nn.Module):
    def __init__(self):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(28*28, 300)
        self.fc2 = nn.Linear(300, 100)
        self.fc3 = nn.Linear(100, 10)
 
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
 
# Training loop
def train(model, loader, optimizer, criterion):
    model.train()
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
 
# Evaluation function
def evaluate(model, loader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
    return correct / len(loader.dataset)
 
# Train the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleMLP().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()
 
print("🚀 Training baseline model...")
for epoch in range(3):
    train(model, train_loader, optimizer, criterion)
    acc = evaluate(model, test_loader)
    print(f"Epoch {epoch+1} - Accuracy: {acc:.2%}")
 
# Apply unstructured pruning (50%) to layers
parameters_to_prune = (
    (model.fc1, 'weight'),
    (model.fc2, 'weight'),
    (model.fc3, 'weight'),
)
for module, param in parameters_to_prune:
    prune.l1_unstructured(module, name=param, amount=0.5)
 
# Remove reparameterization to finalize pruning
for module, param in parameters_to_prune:
    prune.remove(module, param)
 
print("✂️ Model pruned (50% of weights in each layer).")
 
# Evaluate pruned model
pruned_acc = evaluate(model, test_loader)
print(f"🧠 Accuracy after pruning: {pruned_acc:.2%}")