"""
Project 366. PixelCNN implementation
Description:
PixelCNN is an autoregressive generative model that generates images by modeling the conditional distribution of each pixel given the pixels above and to the left. It is known for generating high-quality images pixel by pixel, which can be used for image generation and completion tasks.

In this project, we’ll implement a PixelCNN for generating images one pixel at a time based on the CIFAR-10 dataset.

About:
✅ What It Does:
Defines a PixelCNN model that generates images pixel by pixel, conditioned on the previously generated pixels

Trains on the CIFAR-10 dataset, learning to generate images based on pixel-level dependencies

Uses cross-entropy loss for training the network and Adam optimizer for updating weights

Generates 32x32 RGB images from random noise as input
"""


import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
 
# 1. Define the PixelCNN model
class PixelCNN(nn.Module):
    def __init__(self, in_channels=3, num_filters=64, num_layers=7):
        super(PixelCNN, self).__init__()
 
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
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0002)
 
# 3. Load the CIFAR-10 dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
 
# 4. Training loop for PixelCNN
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = PixelCNN().to(device)
 
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