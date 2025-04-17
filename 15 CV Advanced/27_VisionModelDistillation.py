"""
Project 587: Vision Model Distillation
Description:
Model distillation is a technique in which a smaller model (the "student") is trained to replicate the behavior of a larger, pre-trained model (the "teacher"). This is often used to compress large models for deployment in resource-constrained environments while maintaining performance. In this project, we will implement vision model distillation using a ResNet teacher model and a smaller student model.
"""

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
 
# 1. Define the teacher model (a larger pre-trained model, e.g., ResNet50)
teacher_model = torchvision.models.resnet50(pretrained=True)
teacher_model.eval()  # Set the teacher model to evaluation mode
 
# 2. Define the student model (a smaller model, e.g., ResNet18)
student_model = torchvision.models.resnet18(pretrained=False, num_classes=10)  # CIFAR-10 has 10 classes
student_model.eval()  # Set the student model to evaluation mode
 
# 3. Load CIFAR-10 dataset
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
 
train_data = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
 
# 4. Define the distillation loss function (MSE between teacher and student outputs)
def distillation_loss(student_output, teacher_output, temperature=3):
    loss = nn.KLDivLoss()(nn.functional.log_softmax(student_output / temperature, dim=1),
                          nn.functional.softmax(teacher_output / temperature, dim=1))
    return loss
 
# 5. Initialize optimizer and set the model for distillation
optimizer = torch.optim.Adam(student_model.parameters(), lr=0.001)
 
# 6. Training loop for distillation
def train_distillation(teacher_model, student_model, train_loader, optimizer, num_epochs=5):
    for epoch in range(num_epochs):
        student_model.train()
        total_loss = 0
        
        for images, labels in train_loader:
            optimizer.zero_grad()
 
            # Teacher model prediction (freeze teacher model weights)
            with torch.no_grad():
                teacher_outputs = teacher_model(images)
 
            # Student model prediction
            student_outputs = student_model(images)
 
            # Compute the distillation loss
            loss = distillation_loss(student_outputs, teacher_outputs)
            loss.backward()
            optimizer.step()
 
            total_loss += loss.item()
 
        print(f"Epoch {epoch+1}, Distillation Loss: {total_loss / len(train_loader)}")
 
# 7. Train the student model using distillation
train_distillation(teacher_model, student_model, train_loader, optimizer)
 
# 8. Visualize a sample image from the CIFAR-10 dataset
image = images[0].permute(1, 2, 0).detach().numpy()
plt.imshow(image)
plt.title("Sample Image from CIFAR-10")
plt.show()