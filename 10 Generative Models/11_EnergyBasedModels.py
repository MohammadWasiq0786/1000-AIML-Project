"""
Project 371. Energy-based models
Description:
Energy-based Models (EBMs) are a class of generative models where the model learns an energy function that assigns lower energy to real data and higher energy to generated data. The goal is to model the distribution of data by minimizing the energy for real data and maximizing the energy for fake data. EBMs can be used for both generative modeling and discriminative tasks.

In this project, we’ll implement an Energy-based Model (EBM) for image generation, learning to assign lower energy to real images and higher energy to generated images.

About:
✅ What It Does:
Defines an Energy-based Model (EBM) using convolutional layers to compute an energy score for each image

The model learns to assign lower energy to real images and higher energy to fake images generated from random noise

Trains on the CIFAR-10 dataset, using binary cross-entropy loss to minimize energy for real images and maximize energy for fake images

The model can generate high-quality images by learning the data distribution
"""


import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
 
# 1. Define the Energy-based Model (EBM)
class EnergyBasedModel(nn.Module):
    def __init__(self, in_channels=3, num_filters=64, num_layers=4):
        super(EnergyBasedModel, self).__init__()
 
        self.num_layers = num_layers
        self.conv_layers = nn.ModuleList()
 
        # First convolutional layer with padding
        self.conv_layers.append(nn.Conv2d(in_channels, num_filters, kernel_size=3, stride=1, padding=1))
 
        # Additional convolutional layers
        for _ in range(num_layers - 1):
            self.conv_layers.append(nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding=1))
 
        # Final output layer for energy score
        self.fc = nn.Linear(num_filters * 32 * 32, 1)
 
    def forward(self, x):
        for layer in self.conv_layers:
            x = torch.relu(layer(x))  # Apply ReLU activation
        x = x.view(x.size(0), -1)  # Flatten the image
        return self.fc(x)  # Return energy score
 
# 2. Define the loss function and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0002)
 
# 3. Load the CIFAR-10 dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
 
# 4. Training loop for Energy-based Model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = EnergyBasedModel().to(device)
 
num_epochs = 50
for epoch in range(num_epochs):
    for i, (real_images, _) in enumerate(train_loader):
        real_images = real_images.to(device)
 
        # Forward pass
        optimizer.zero_grad()
        energy_real = model(real_images)
 
        # Generate fake images using random noise
        fake_images = torch.randn(real_images.size()).to(device)
        energy_fake = model(fake_images)
 
        # Compute the loss (minimize energy for real images, maximize energy for fake images)
        loss = criterion(energy_real, torch.ones_like(energy_real)) + criterion(energy_fake, torch.zeros_like(energy_fake))
        loss.backward()
 
        # Update the model weights
        optimizer.step()
 
    # Print loss every 10 epochs
    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
 
    # Generate and display sample images
    if (epoch+1) % 10 == 0:
        with torch.no_grad():
            fake_images = torch.randn(64, 3, 32, 32).to(device)
            grid_img = torchvision.utils.make_grid(fake_images, nrow=8, normalize=True)
            plt.imshow(grid_img.permute(1, 2, 0))
            plt.show()