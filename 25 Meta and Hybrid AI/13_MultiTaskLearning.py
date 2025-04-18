"""
Project 973: Multi-task Learning Implementation
Description
Multi-task learning (MTL) involves training a model to perform multiple tasks simultaneously, sharing common representations between tasks. In this project, we will implement a multi-task learning framework to classify images while also predicting other attributes, such as age or gender, from the same image.

Key Concepts Covered:
Multi-task Learning (MTL): A single model is trained to perform multiple tasks simultaneously, sharing common layers while having task-specific branches.

Shared Representation: The model learns a shared representation in the early layers, while separate output layers are dedicated to each task.

Task-specific Losses: Each task has its own loss function (e.g., cross-entropy loss for classification and mean squared error for regression).
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
 
# Define the multi-task learning model
class MultiTaskModel(nn.Module):
    def __init__(self):
        super(MultiTaskModel, self).__init__()
        self.resnet = models.resnet50(pretrained=True)  # Pre-trained ResNet50
        
        # Replace the final fully connected layer with a new one for shared feature extraction
        self.resnet.fc = nn.Identity()  # Remove the original fully connected layer
 
        # Task-specific branches
        self.classifier = nn.Linear(self.resnet.fc.in_features, 10)  # CIFAR-10 classification (10 classes)
        self.age_predictor = nn.Linear(self.resnet.fc.in_features, 1)  # Age prediction (1 output)
        self.gender_predictor = nn.Linear(self.resnet.fc.in_features, 2)  # Gender prediction (2 classes)
 
    def forward(self, x):
        # Extract features from the ResNet backbone
        features = self.resnet(x)
        
        # Task-specific predictions
        class_output = self.classifier(features)
        age_output = self.age_predictor(features)
        gender_output = self.gender_predictor(features)
        
        return class_output, age_output, gender_output
 
# Initialize the model, optimizer, and loss function
model = MultiTaskModel()
optimizer = optim.Adam(model.parameters(), lr=0.001)
 
# Define loss functions for each task
classification_loss_fn = nn.CrossEntropyLoss()
regression_loss_fn = nn.MSELoss()
 
# Training the multi-task model
for epoch in range(5):  # Multi-task training for 5 epochs
    model.train()
    total_loss = 0
    for data, target in train_loader:
        optimizer.zero_grad()
 
        # Get predictions for each task
        class_output, age_output, gender_output = model(data)
        
        # Task-specific losses
        class_loss = classification_loss_fn(class_output, target)  # Classification loss
        age_loss = regression_loss_fn(age_output, target.float().view(-1, 1))  # Age regression loss
        gender_loss = classification_loss_fn(gender_output, target)  # Gender classification loss
        
        # Total loss as the sum of all task losses
        loss = class_loss + age_loss + gender_loss
        loss.backward()
        optimizer.step()
 
        total_loss += loss.item()
 
    print(f"Epoch {epoch+1}, Loss: {total_loss / len(train_loader)}")
 
# Evaluate the model on the test dataset
model.eval()
correct_classifications, correct_gender, total = 0, 0, 0
total_age_error = 0
 
with torch.no_grad():
    for data, target in test_loader:
        class_output, age_output, gender_output = model(data)
 
        # Classification accuracy
        _, predicted_class = torch.max(class_output, 1)
        correct_classifications += (predicted_class == target).sum().item()
 
        # Gender classification accuracy
        _, predicted_gender = torch.max(gender_output, 1)
        correct_gender += (predicted_gender == target).sum().item()
 
        # Age prediction error (MSE)
        total_age_error += torch.abs(age_output - target.float().view(-1, 1)).sum().item()
 
        total += target.size(0)
 
# Print results
print(f"Classification Accuracy: {100 * correct_classifications / total:.2f}%")
print(f"Gender Classification Accuracy: {100 * correct_gender / total:.2f}%")
print(f"Average Age Prediction Error: {total_age_error / total:.2f}")