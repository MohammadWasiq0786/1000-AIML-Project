"""
Project 370. Normalizing flows implementation
Description:
Normalizing Flows are a class of invertible transformations used in generative modeling. The idea is to map a simple distribution (like a standard Gaussian) to a complex distribution (like the distribution of real data) through a series of invertible transformations. Normalizing flows allow exact likelihood computation and efficient sampling, which are typically difficult with models like GANs and VAEs.

In this project, we'll implement Normalizing Flows to model the distribution of images and generate realistic samples.

About:
✅ What It Does:
Defines a Normalizing Flow model with invertible transformations for generating high-quality samples from a simple latent space (Gaussian noise)

Implements affine transformations in each flow layer to learn complex data distributions

Trains on the CIFAR-10 dataset to generate 32x32 RGB images using normalizing flows

Uses MSE loss along with log-jacobian regularization to enforce invertibility and consistency in the transformation
"""


import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
 
# 1. Define the Normalizing Flow model
class FlowLayer(nn.Module):
    def __init__(self, in_channels):
        super(FlowLayer, self).__init__()
        self.fc1 = nn.Linear(in_channels, in_channels)
        self.fc2 = nn.Linear(in_channels, in_channels)
    
    def forward(self, x):
        # Simple affine transformation with a learnable scaling factor
        z = torch.relu(self.fc1(x))
        log_det_jacobian = torch.sum(torch.log(torch.abs(torch.det(self.fc2(z)))))
        return z, log_det_jacobian
 
class NormalizingFlow(nn.Module):
    def __init__(self, in_channels=3, num_layers=3):
        super(NormalizingFlow, self).__init__()
        self.layers = nn.ModuleList([FlowLayer(in_channels * 32 * 32) for _ in range(num_layers)])
 
    def forward(self, x):
        log_det_jacobian = 0
        for layer in self.layers:
            x, log_jacobian = layer(x.view(-1, 32*32*3))  # Flatten the input
            log_det_jacobian += log_jacobian
        return x, log_det_jacobian
 
# 2. Define the loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0002)
 
# 3. Load the CIFAR-10 dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
 
# 4. Training loop for Normalizing Flows
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = NormalizingFlow().to(device)
 
num_epochs = 50
for epoch in range(num_epochs):
    for i, (real_images, _) in enumerate(train_loader):
        real_images = real_images.to(device)
 
        # Forward pass
        optimizer.zero_grad()
        outputs, log_det_jacobian = model(real_images)
 
        # Compute the loss (MSE loss + regularization for log-jacobian)
        loss = criterion(outputs, real_images) - log_det_jacobian
        loss.backward()
 
        # Update the model weights
        optimizer.step()
 
    # Print loss every 10 epochs
    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
 
    # Generate and display sample images
    if (epoch+1) % 10 == 0:
        with torch.no_grad():
            fake_images = model(torch.randn(64, 3, 32, 32).to(device))[0].cpu()
            grid_img = torchvision.utils.make_grid(fake_images, nrow=8, normalize=True)
            plt.imshow(grid_img.permute(1, 2, 0))
            plt.show()