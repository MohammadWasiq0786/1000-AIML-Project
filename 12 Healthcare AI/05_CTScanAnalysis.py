"""
Project 445. CT scan analysis
Description:
CT (Computed Tomography) scans provide cross-sectional images of the body and are critical for diagnosing conditions like tumors, lesions, COVID-19, and organ abnormalities. In this project, we’ll build a deep learning model to classify CT scan slices as normal or diseased (e.g., COVID-positive vs negative), using a CNN-based classifier.

About:
✅ What It Does:
Simulates CT scan slices and builds a classifier for binary disease classification.

Uses ResNet18, fine-tuned for CT image inputs.

Can be extended for:

3D CT volume classification

Lesion segmentation

Explainability (Grad-CAM)

COVID-CT: https://github.com/UCSD-AI4H/COVID-CT

MosMedData: COVID-19 CT scans
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
 
# 1. Simulated CT scan transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),  # CTs are usually grayscale
    transforms.ToTensor()
])
 
# 2. Fake CT dataset (replace with real CT scan slices)
train_data = datasets.FakeData(size=100, image_size=(3, 224, 224), num_classes=2, transform=transform)
test_data = datasets.FakeData(size=20, image_size=(3, 224, 224), num_classes=2, transform=transform)
train_loader = DataLoader(train_data, batch_size=8, shuffle=True)
test_loader = DataLoader(test_data, batch_size=8)
 
# 3. Define CT classifier using pretrained ResNet18
class CTScanClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.resnet18(pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, 2)
 
    def forward(self, x):
        return self.model(x)
 
# 4. Training setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CTScanClassifier().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()
 
# 5. Train loop
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
    correct = total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            preds = model(images).argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    print(f"Test Accuracy: {correct / total:.2f}")
 
# 7. Train model
for epoch in range(1, 6):
    train()
    print(f"Epoch {epoch}")
    evaluate()