"""
Project 136. PixelCNN implementation
Description:
PixelCNN is an autoregressive generative model for images that models the conditional distribution of each pixel given the previous pixels. It generates images pixel-by-pixel, respecting spatial dependencies. In this project, we implement a simplified PixelCNN using masked convolutions to ensure the model only "sees" previous pixels during training.

About:
🧠 What This Project Demonstrates:
Implements Masked Convolutions to preserve autoregressive structure

Trains a PixelCNN to model conditional pixel distributions

Uses CrossEntropy loss to predict pixel intensity values (0–255)
"""

# Install if not already: pip install torch torchvision matplotlib
 
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
 
# Define Masked Convolution Layer
class MaskedConv2d(nn.Conv2d):
    def __init__(self, mask_type, *args, **kwargs):
        super(MaskedConv2d, self).__init__(*args, **kwargs)
        assert mask_type in {'A', 'B'}
        self.register_buffer("mask", torch.ones_like(self.weight.data))
        
        _, _, h, w = self.weight.size()
        yc, xc = h // 2, w // 2
        self.mask[:, :, yc, xc + (mask_type == 'B'):] = 0
        self.mask[:, :, yc + 1:] = 0
 
    def forward(self, x):
        self.weight.data *= self.mask
        return super(MaskedConv2d, self).forward(x)
 
# Simple PixelCNN model
class PixelCNN(nn.Module):
    def __init__(self, input_channels=1, n_filters=64, kernel_size=7, n_layers=7):
        super(PixelCNN, self).__init__()
        layers = [MaskedConv2d('A', input_channels, n_filters, kernel_size, padding=kernel_size // 2)]
        layers.append(nn.ReLU())
        for _ in range(n_layers - 2):
            layers.append(MaskedConv2d('B', n_filters, n_filters, kernel_size, padding=kernel_size // 2))
            layers.append(nn.ReLU())
        layers.append(MaskedConv2d('B', n_filters, 256, 1))  # Final layer predicts pixel intensities
        self.net = nn.Sequential(*layers)
 
    def forward(self, x):
        return self.net(x)
 
# Load MNIST data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: (x * 255).long())  # Convert pixel values to [0, 255]
])
dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
 
# Model, loss, optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = PixelCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
 
# One training step (demo only)
model.train()
for i, (imgs, _) in enumerate(dataloader):
    imgs = imgs.to(device)
    targets = imgs.squeeze(1)
    outputs = model(imgs.float() / 255.0)  # Normalize input
    loss = criterion(outputs, targets)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    print(f"PixelCNN Training Loss (1 batch): {loss.item():.4f}")
    break