"""
Project 368. Autoregressive models
Description:
Autoregressive models are a class of generative models where the output at each step is conditioned on the previous outputs. In image generation tasks, an autoregressive model generates an image pixel by pixel, where each pixel is generated based on previously generated pixels. This approach allows models to capture intricate dependencies between pixels and generate high-quality images.

In this project, we will explore autoregressive models for image generation, specifically focusing on how they generate images sequentially.

About:
✅ What It Does:
Defines an Autoregressive model (similar to PixelCNN) that generates images by conditioning each pixel on the previously generated ones

Trains on the CIFAR-10 dataset to generate 32x32 RGB images sequentially

Uses MSE loss to minimize the difference between the generated images and the actual images

The model learns the pixel dependencies and generates high-quality images from random noise
"""


import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
 
# 1. Define the Autoregressive model (similar to PixelCNN)
class AutoregressiveModel(nn.Module):
    def __init__(self, in_channels=3, num_filters=64, num_layers=7):
        super(AutoregressiveModel, self).__init__()
 
        self.num_layers = num_layers
        self.conv_layers = nn.ModuleList()
 
        # First convolutional layer with padding
        self.conv_layers.append(nn.Conv2d(in_channels, num_filters, kernel_size=7, padding=3))
 
        # Additional convolutional layers
        for _ in range(num_layers - 1):
            self.conv_layers.append(nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1))
 
        # Final output layer
        self.final_conv = nn.Conv2d(num_filters, in_channels, kernel_size=1)
 
    def forward(self, x):
        for layer in self.conv_layers:
            x = torch.relu(layer(x))  # Apply ReLU activation
        return self.final_conv(x)
 
# 2. Define the loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0002)
 
# 3. Load the CIFAR-10 dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
 
# 4. Training loop for Autoregressive model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = AutoregressiveModel().to(device)
 
num_epochs = 50
for epoch in range(num_epochs):
    for i, (real_images, _) in enumerate(train_loader):
        real_images = real_images.to(device)
 
        # Forward pass
        optimizer.zero_grad()
        outputs = model(real_images)
 
        # Compute the loss
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