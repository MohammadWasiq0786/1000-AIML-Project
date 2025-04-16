"""
Project 444. X-ray analysis system
Description:
An X-ray analysis system uses deep learning to detect abnormalities such as pneumonia, fractures, or COVID-19 from chest or limb X-ray images. In this project, we'll build a system that classifies chest X-rays as normal or pneumonia-positive using a pre-trained CNN (ResNet) fine-tuned on X-ray data.

About:
✅ What It Does:
Loads synthetic or real chest X-ray images.

Fine-tunes a ResNet18 to classify X-rays into normal vs pneumonia.

Can be extended with Grad-CAM for explainability and multi-label classification for multiple diseases.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
 
# 1. Define transform for X-ray images
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),  # convert to 3-channel
    transforms.ToTensor()
])
 
# 2. Simulate X-ray image data (use real dataset in production)
train_data = datasets.FakeData(size=100, image_size=(3, 224, 224), num_classes=2, transform=transform)
test_data = datasets.FakeData(size=20, image_size=(3, 224, 224), num_classes=2, transform=transform)
train_loader = DataLoader(train_data, batch_size=8, shuffle=True)
test_loader = DataLoader(test_data, batch_size=8)
 
# 3. Pre-trained ResNet18 for binary classification
class XrayClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = models.resnet18(pretrained=True)
        self.base.fc = nn.Linear(self.base.fc.in_features, 2)
 
    def forward(self, x):
        return self.base(x)
 
# 4. Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = XrayClassifier().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()
 
# 5. Training loop
def train():
    model.train()
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
 
# 6. Evaluation
def evaluate():
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            preds = torch.argmax(model(images), dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    print(f"Test Accuracy: {correct / total:.2f}")
 
# 7. Run training
for epoch in range(1, 6):
    train()
    print(f"Epoch {epoch}")
    evaluate()