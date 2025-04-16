"""
Project 442. Medical image classification
Description:
This project builds a deep learning model to classify medical images into diagnostic categories — for example, identifying whether an X-ray indicates pneumonia. It can be used in radiology support tools, triage systems, or clinical decision support.

We’ll simulate it using a subset of chest X-ray images and a CNN-based classifier.

About:
✅ What It Does:
Loads simulated (or real) medical images.

Uses a ResNet18 CNN to classify images into 2 categories (e.g., normal vs pneumonia).

Trains and evaluates on simple data (for demo).

Can easily scale to real medical datasets.
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torchvision.models as models
 
# 1. Simulate medical images (replace with real data for production)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])
 
train_data = datasets.FakeData(size=100, image_size=(3, 224, 224), num_classes=2, transform=transform)
test_data = datasets.FakeData(size=20, image_size=(3, 224, 224), num_classes=2, transform=transform)
 
train_loader = DataLoader(train_data, batch_size=8, shuffle=True)
test_loader = DataLoader(test_data, batch_size=8)
 
# 2. Define CNN model (ResNet18 fine-tuned)
class MedicalCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = models.resnet18(pretrained=True)
        self.base.fc = nn.Linear(self.base.fc.in_features, 2)  # binary classification
 
    def forward(self, x):
        return self.base(x)
 
# 3. Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MedicalCNN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()
 
# 4. Training function
def train():
    model.train()
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
 
# 5. Evaluation function
def evaluate():
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    print(f"Test Accuracy: {correct / total:.2f}")
 
# 6. Run training
for epoch in range(1, 6):
    train()
    print(f"Epoch {epoch}")
    evaluate()