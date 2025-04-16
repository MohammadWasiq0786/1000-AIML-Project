"""
Project 158. Adversarial attack generator
Description:
Adversarial attacks involve subtly modifying input data to fool a machine learning model. These changes are often imperceptible to humans but can cause incorrect predictions. In this project, we implement a basic Fast Gradient Sign Method (FGSM) attack to generate adversarial examples that trick a trained classifier (e.g., MNIST CNN).

About:
🧠 What This Project Demonstrates:
Creates adversarial examples using FGSM (Fast Gradient Sign Method)

Shows how small pixel perturbations can fool a classifier

Visualizes and compares original vs adversarial images
"""

# Install if not already: pip install torch torchvision matplotlib
 
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
 
# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 
# Load MNIST
transform = transforms.ToTensor()
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)
 
# Simple CNN for MNIST (already trained or just for demo)
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
 
# Load model (assume trained)
model = SimpleCNN().to(device)
model.load_state_dict(torch.load("mnist_cnn.pth"))  # Load your pre-trained weights
model.eval()
 
# FGSM Attack Function
def fgsm_attack(image, label, epsilon):
    image.requires_grad = True
    output = model(image)
    loss = F.nll_loss(F.log_softmax(output, dim=1), label)
    model.zero_grad()
    loss.backward()
    data_grad = image.grad.data
    perturbed_image = image + epsilon * data_grad.sign()
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    return perturbed_image
 
# Run a single adversarial attack
epsilon = 0.25
for data, target in test_loader:
    data, target = data.to(device), target.to(device)
    original_pred = model(data).argmax(dim=1).item()
 
    adv_data = fgsm_attack(data, target, epsilon)
    adv_pred = model(adv_data).argmax(dim=1).item()
 
    print(f"🎯 Original Prediction: {original_pred}")
    print(f"⚠️ Adversarial Prediction: {adv_pred}")
 
    # Display original vs adversarial image
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.title("Original")
    plt.imshow(data.squeeze().cpu().numpy(), cmap="gray")
    plt.axis("off")
 
    plt.subplot(1, 2, 2)
    plt.title("Adversarial")
    plt.imshow(adv_data.squeeze().detach().cpu().numpy(), cmap="gray")
    plt.axis("off")
    plt.show()
    break