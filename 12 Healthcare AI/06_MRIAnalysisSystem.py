"""
Project 446. MRI analysis system
Description:
MRI (Magnetic Resonance Imaging) is widely used to visualize soft tissues, especially in the brain and spine. AI models can analyze MRI slices to detect tumors, stroke lesions, or neurodegenerative diseases. In this project, we’ll create a CNN classifier to analyze MRI images and classify them as normal or abnormal.

✅ What It Does:
Simulates a binary classification model for MRI image analysis.

Fine-tunes a ResNet18 on simulated brain MRI scans.

Easily extendable for:

Multi-class tumor classification

Segmentation of lesions/tumors

3D MRI volume analysis (using 3D CNNs)

Brain Tumor MRI Dataset (Kaggle)

BRATS (Brain Tumor Segmentation Challenge)
"""


import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
 
# 1. Transform for grayscale MRI scans
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),  # convert grayscale to 3-channel for pretrained model
    transforms.ToTensor()
])
 
# 2. Simulated MRI dataset (use real data in production)
train_data = datasets.FakeData(size=100, image_size=(3, 224, 224), num_classes=2, transform=transform)
test_data = datasets.FakeData(size=20, image_size=(3, 224, 224), num_classes=2, transform=transform)
train_loader = DataLoader(train_data, batch_size=8, shuffle=True)
test_loader = DataLoader(test_data, batch_size=8)
 
# 3. Define MRI classifier (ResNet18 fine-tuned)
class MRINet(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = models.resnet18(pretrained=True)
        self.base.fc = nn.Linear(self.base.fc.in_features, 2)
 
    def forward(self, x):
        return self.base(x)
 
# 4. Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MRINet().to(device)
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