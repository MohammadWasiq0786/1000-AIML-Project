"""
Project 372. Diffusion models implementation
Description:
Diffusion models are a class of generative models that learn to transform noise into structured data, such as images, through a process of gradual refinement. The model learns to reverse a noising process, where data is progressively corrupted by noise. It then learns how to reverse this process to generate data from noise. This approach has gained popularity for generating high-quality images and is known for its flexibility and sharpness in image generation.

In this project, we’ll implement a Diffusion Model to generate images by reversing the diffusion process.

About:
✅ What It Does:
Defines a Diffusion model that generates images by reversing the noising process

The forward process adds noise to images, and the model learns to reverse this process to recover the original image

Uses Mean Squared Error (MSE) loss to compare the generated image with the original image

Trains on the CIFAR-10 dataset to generate 32x32 RGB images

The model learns to generate images from random noise by progressively refining them
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
 
# 1. Define the Diffusion model
class DiffusionModel(nn.Module):
    def __init__(self, in_channels=3, num_filters=64):
        super(DiffusionModel, self).__init__()
 
        self.fc1 = nn.Linear(32 * 32 * 3, num_filters)
        self.fc2 = nn.Linear(num_filters, num_filters * 2)
        self.fc3 = nn.Linear(num_filters * 2, 32 * 32 * 3)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
 
    def forward(self, x):
        x = x.view(-1, 32 * 32 * 3)  # Flatten the image
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.tanh(self.fc3(x))
        return x.view(-1, 3, 32, 32)  # Reshape back to image dimensions
 
# 2. Define the loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0002)
 
# 3. Load the CIFAR-10 dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
 
# 4. Diffusion process for noise addition (forward process)
def add_noise(images, noise_factor=0.5):
    noisy_images = images + noise_factor * torch.randn_like(images)
    noisy_images = torch.clamp(noisy_images, 0., 1.)
    return noisy_images
 
# 5. Training loop for Diffusion Model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = DiffusionModel().to(device)
 
num_epochs = 50
for epoch in range(num_epochs):
    for i, (real_images, _) in enumerate(train_loader):
        real_images = real_images.to(device)
 
        # Add noise to the images (diffusion process)
        noisy_images = add_noise(real_images)
 
        # Forward pass
        optimizer.zero_grad()
        outputs = model(noisy_images)
 
        # Compute the loss (Mean Squared Error between generated and real images)
        loss = criterion(outputs, real_images)
        loss.backward()
 
        # Update the model weights
        optimizer.step()
 
    # Print loss every 10 epochs
    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
 
    # Generate and display sample images
    if (epoch+1) % 10 == 0:
        with torch.no_grad():
            fake_images = model(torch.randn(64, 3, 32, 32).to(device)).cpu()
            grid_img = torchvision.utils.make_grid(fake_images, nrow=8, normalize=True)
            plt.imshow(grid_img.permute(1, 2, 0))
            plt.show()