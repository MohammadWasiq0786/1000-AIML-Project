"""
Project 972: Transfer Learning Framework
Description
Transfer learning involves leveraging knowledge from one domain (usually with a large amount of data) and applying it to another domain (usually with limited data). In this project, we will implement a transfer learning framework using pre-trained models and fine-tune them for a different task or domain.

Key Concepts Covered:
Transfer Learning: Using a model pre-trained on a large dataset (e.g., ImageNet) and adapting it to a new, smaller dataset (e.g., CIFAR-10).

Fine-Tuning: Freezing the layers of a pre-trained model and training only the final layers for a new task.

Pre-trained Models: Models that have been trained on large datasets and can be used as feature extractors for new tasks.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
 
# Load CIFAR-10 dataset
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resizing to fit ResNet50 input size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Pre-trained model normalization
])
 
train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
 
# Split the dataset into train and validation sets
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
 
# Load pre-trained ResNet50 model
model = models.resnet50(pretrained=True)
 
# Freeze all layers except the final fully connected layer
for param in model.parameters():
    param.requires_grad = False
 
# Modify the final fully connected layer to match CIFAR-10 classes (10 classes)
model.fc = nn.Linear(model.fc.in_features, 10)
 
# Use Adam optimizer and cross-entropy loss
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()
 
# Train the model on CIFAR-10 using transfer learning
model.train()
for epoch in range(5):  # Fine-tune for 5 epochs
    total_loss = 0
    for data, target in train_loader:
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss / len(train_loader)}")
 
# Test the fine-tuned model
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for data, target in test_loader:
        output = model(data)
        _, predicted = torch.max(output, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()
 
print(f"Accuracy on test set: {100 * correct / total:.2f}%")