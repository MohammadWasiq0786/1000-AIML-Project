"""
Project 447. Skin cancer detection
Description:
Skin cancer, including melanoma, is one of the most common and deadly cancers. AI models can assist in early detection by analyzing dermatology images of skin lesions. In this project, we’ll build a deep learning classifier to detect whether a skin lesion is benign or malignant, using a CNN model.

✅ What It Does:
Simulates a binary classifier for benign vs malignant lesion classification.

Uses ResNet18, fine-tuned for skin image inputs.

Can be extended to:

Handle multi-class skin conditions

Add Grad-CAM to show model attention

Deploy on mobile apps or teledermatology platforms

You can replace the dummy dataset with real datasets like:

ISIC Skin Lesion Dataset (International Skin Imaging Collaboration)

HAM10000 dataset on Kaggle
"""

import torch
import torch.nn as nn
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader
 
# 1. Image transform for dermoscopic skin lesion images
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])  # normalize grayscale or RGB
])
 
# 2. Simulated dataset (use ISIC/HAM10000 for real case)
train_data = datasets.FakeData(size=100, image_size=(3, 224, 224), num_classes=2, transform=transform)
test_data = datasets.FakeData(size=20, image_size=(3, 224, 224), num_classes=2, transform=transform)
train_loader = DataLoader(train_data, batch_size=8, shuffle=True)
test_loader = DataLoader(test_data, batch_size=8)
 
# 3. CNN model for binary classification
class SkinCancerCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = models.resnet18(pretrained=True)
        self.base.fc = nn.Linear(self.base.fc.in_features, 2)
 
    def forward(self, x):
        return self.base(x)
 
# 4. Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SkinCancerCNN().to(device)
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
    correct = total = 0
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