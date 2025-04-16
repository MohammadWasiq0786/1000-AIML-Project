"""
Project 448. Diabetic retinopathy detection
Description:
Diabetic Retinopathy (DR) is a diabetes complication that affects the eyes and can cause blindness if not detected early. AI models can analyze retinal fundus images to detect DR and classify its severity stages. In this project, we build a CNN-based classifier to predict whether a retina scan shows signs of DR.

✅ What It Does:
Classifies simulated retinal images into DR-positive or normal.

Uses a fine-tuned ResNet18 CNN for high accuracy.

Can be extended to:

Multi-class grading of DR (from 0 to 4)

Image enhancement preprocessing

Attention heatmaps for clinical transparency

You can replace the dummy dataset with real datasets like:

APTOS 2019 Blindness Detection (Kaggle)

EyePACS Dataset
"""

import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
 
# 1. Retina image transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])  # normalize fundus image data
])
 
# 2. Simulated retina image dataset (use APTOS or EyePACS in production)
train_data = datasets.FakeData(size=100, image_size=(3, 224, 224), num_classes=2, transform=transform)
test_data = datasets.FakeData(size=20, image_size=(3, 224, 224), num_classes=2, transform=transform)
train_loader = DataLoader(train_data, batch_size=8, shuffle=True)
test_loader = DataLoader(test_data, batch_size=8)
 
# 3. CNN model for binary classification (DR vs No DR)
class DRClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = models.resnet18(pretrained=True)
        self.base.fc = nn.Linear(self.base.fc.in_features, 2)
 
    def forward(self, x):
        return self.base(x)
 
# 4. Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DRClassifier().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()
 
# 5. Training function
def train():
    model.train()
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
 
# 6. Evaluation function
def evaluate():
    model.eval()
    correct, total = 0, 0
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