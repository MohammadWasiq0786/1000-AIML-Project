"""
Project 979: Knowledge Distillation Framework
Description
Knowledge distillation is a technique where a large, complex model (teacher) is used to train a smaller, more efficient model (student). The student model learns from the teacher's predictions, effectively transferring the knowledge while maintaining the same performance but with fewer resources. In this project, we will implement a knowledge distillation framework for a teacher-student setup in image classification.

Key Concepts Covered:
Teacher-Student Models: Knowledge distillation involves training a smaller (student) model using the output of a larger (teacher) model.

Distillation Loss: The loss function combines the standard classification loss (cross-entropy) and a KL divergence between the student and teacher's soft outputs (after applying a temperature scaling).

Temperature Scaling: In distillation, the softmax function is applied with a higher temperature to soften the output probabilities of the teacher model, allowing the student model to learn from more nuanced information.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import numpy as np
 
# Define a simple CNN for the student model (smaller architecture)
class StudentCNN(nn.Module):
    def __init__(self):
        super(StudentCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3)
        self.fc1 = nn.Linear(32 * 6 * 6, 128)
        self.fc2 = nn.Linear(128, 10)
 
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
 
# Define the teacher model (ResNet50)
teacher_model = models.resnet50(pretrained=True)
teacher_model.fc = nn.Linear(teacher_model.fc.in_features, 10)  # CIFAR-10 has 10 classes
 
# Define the distillation loss function (combines cross-entropy and KL divergence)
class DistillationLoss(nn.Module):
    def __init__(self, T=2, alpha=0.5):
        super(DistillationLoss, self).__init__()
        self.T = T  # Temperature parameter for softening the teacher's outputs
        self.alpha = alpha  # Weight for the distillation loss
 
    def forward(self, student_output, teacher_output, true_labels):
        # Soft labels (teacher's predictions with softmax)
        soft_teacher_output = torch.softmax(teacher_output / self.T, dim=1)
        soft_student_output = torch.softmax(student_output / self.T, dim=1)
 
        # Cross-entropy loss for hard labels (true labels)
        hard_loss = nn.CrossEntropyLoss()(student_output, true_labels)
 
        # KL divergence between student and teacher's soft labels
        soft_loss = nn.KLDivLoss()(torch.log(soft_student_output), soft_teacher_output)
 
        # Combine the losses: alpha * soft_loss + (1 - alpha) * hard_loss
        loss = self.alpha * soft_loss + (1 - self.alpha) * hard_loss
        return loss
 
# Load CIFAR-10 dataset
transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
 
# Initialize the student model, optimizer, and loss function
student_model = StudentCNN()
optimizer = optim.Adam(student_model.parameters(), lr=0.001)
distillation_loss_fn = DistillationLoss()
 
# Train the student model using knowledge distillation
for epoch in range(5):  # Training for 5 epochs
    student_model.train()
    teacher_model.eval()  # Teacher is fixed (not trained)
    total_loss = 0
    for data, target in train_loader:
        optimizer.zero_grad()
 
        # Get the teacher's predictions (soft labels)
        with torch.no_grad():
            teacher_output = teacher_model(data)
 
        # Get the student's predictions
        student_output = student_model(data)
 
        # Calculate the distillation loss
        loss = distillation_loss_fn(student_output, teacher_output, target)
        loss.backward()
        optimizer.step()
 
        total_loss += loss.item()
 
    print(f"Epoch {epoch+1}, Loss: {total_loss / len(train_loader)}")
 
# Evaluate the student model on the test set
student_model.eval()
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
 
correct = 0
total = 0
with torch.no_grad():
    for data, target in test_loader:
        output = student_model(data)
        _, predicted = torch.max(output, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()
 
print(f"Student Model Accuracy on Test Data: {100 * correct / total:.2f}%")