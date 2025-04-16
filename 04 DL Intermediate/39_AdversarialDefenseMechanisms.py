"""
Project 159. Adversarial defense mechanisms
Description:
Adversarial defenses aim to make machine learning models robust against adversarial attacks (e.g., FGSM, PGD). In this project, we implement two basic defenses: adversarial training and input preprocessing (e.g., JPEG compression) on an MNIST classifier to counter FGSM attacks.

About:
🧠 What This Project Demonstrates:
Implements adversarial training: model learns from real + adversarial examples

Applies input preprocessing defense (JPEG compression to smooth perturbations)

Shows measurable difference in accuracy under FGSM attack
"""

# Install if not already: pip install torch torchvision pillow
 
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torchvision.transforms.functional as TF
from PIL import Image
import io
 
# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 
# Simple CNN
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 5 * 5, 128), nn.ReLU(),
            nn.Linear(128, 10)
        )
 
    def forward(self, x):
        return self.net(x)
 
# Load dataset
transform = transforms.ToTensor()
train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_dataset = datasets.MNIST('./data', train=False, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)
 
# FGSM Attack
def fgsm_attack(model, data, target, epsilon):
    data.requires_grad = True
    output = model(data)
    loss = F.cross_entropy(output, target)
    model.zero_grad()
    loss.backward()
    data_grad = data.grad.data
    return torch.clamp(data + epsilon * data_grad.sign(), 0, 1)
 
# Input Preprocessing: simple JPEG compression simulation
def jpeg_compression(image_tensor):
    pil_image = TF.to_pil_image(image_tensor.squeeze())
    buffer = io.BytesIO()
    pil_image.save(buffer, format='JPEG', quality=75)
    buffer.seek(0)
    compressed_image = Image.open(buffer)
    return TF.to_tensor(compressed_image).unsqueeze(0)
 
# Adversarial training loop
def train_with_adversarial(model, loader, epsilon=0.2):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(2):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            # Generate adversarial examples on the fly
            adv_images = fgsm_attack(model, images.clone(), labels, epsilon)
            images_combined = torch.cat([images, adv_images])
            labels_combined = torch.cat([labels, labels])
 
            outputs = model(images_combined)
            loss = F.cross_entropy(outputs, labels_combined)
 
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"📘 Epoch {epoch+1} done (Adversarial Training)")
 
# Evaluation with and without preprocessing
def evaluate(model, loader, epsilon=0.2, use_preprocessing=False):
    model.eval()
    correct = 0
    total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        adv_images = fgsm_attack(model, images.clone(), labels, epsilon)
 
        if use_preprocessing:
            adv_images = jpeg_compression(adv_images.cpu()).to(device)
 
        outputs = model(adv_images)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        if total >= 100:  # evaluate on 100 samples
            break
    return correct / total
 
# Initialize and train model
model = SimpleCNN().to(device)
train_with_adversarial(model, train_loader, epsilon=0.2)
 
# Evaluate with and without preprocessing
acc_no_defense = evaluate(model, test_loader, use_preprocessing=False)
acc_with_preprocessing = evaluate(model, test_loader, use_preprocessing=True)
 
print(f"⚠️ Accuracy against adversarial (no defense): {acc_no_defense:.2%}")
print(f"🛡️ Accuracy with JPEG preprocessing defense: {acc_with_preprocessing:.2%}")