"""
Project 153. Knowledge distillation implementation
Description:
Knowledge Distillation is a technique to transfer knowledge from a large, accurate model (teacher) to a smaller, faster model (student). The student learns not just from hard labels but also from the soft predictions of the teacher, enabling better generalization even with fewer parameters. This project demonstrates knowledge distillation using two simple neural networks on the MNIST dataset.

About:
🧠 What This Project Demonstrates:
Builds a teacher-student training pipeline

Uses KL-divergence on soft labels to guide the student

Distills knowledge to a smaller, faster model with competitive accuracy
"""

# Install if not already: pip install torch torchvision matplotlib
 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
 
# Load MNIST
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000)
 
# Define Teacher and Student Networks
class TeacherNet(nn.Module):
    def __init__(self):
        super(TeacherNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
        )
    def forward(self, x):
        return self.net(x.view(x.size(0), -1))
 
class StudentNet(nn.Module):
    def __init__(self):
        super(StudentNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(28*28, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )
    def forward(self, x):
        return self.net(x.view(x.size(0), -1))
 
# Knowledge Distillation Loss
def distillation_loss(student_logits, teacher_logits, labels, T=2.0, alpha=0.7):
    soft_loss = F.kl_div(
        F.log_softmax(student_logits / T, dim=1),
        F.softmax(teacher_logits / T, dim=1),
        reduction='batchmean'
    ) * (T * T)
    hard_loss = F.cross_entropy(student_logits, labels)
    return alpha * soft_loss + (1 - alpha) * hard_loss
 
# Initialize models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
teacher = TeacherNet().to(device)
student = StudentNet().to(device)
 
# Step 1: Pre-train the teacher model
optimizer = torch.optim.Adam(teacher.parameters(), lr=0.001)
teacher.train()
for epoch in range(1):
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        logits = teacher(images)
        loss = F.cross_entropy(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
 
print("🎓 Teacher training done.")
 
# Step 2: Train student with knowledge distillation
optimizer = torch.optim.Adam(student.parameters(), lr=0.001)
student.train()
teacher.eval()  # Freeze teacher
for epoch in range(3):
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        with torch.no_grad():
            teacher_logits = teacher(images)
        student_logits = student(images)
        loss = distillation_loss(student_logits, teacher_logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"📘 Epoch {epoch+1} - Distillation Loss: {loss.item():.4f}")
 
# Step 3: Evaluate student
student.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        preds = student(images).argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
 
print(f"🎯 Student Model Accuracy after Distillation: {correct / total:.2%}")