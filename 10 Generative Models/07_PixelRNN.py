"""
Project 367. PixelRNN implementation
Description:
PixelRNN is a generative model similar to PixelCNN, but instead of using convolutions, it employs recurrent neural networks (RNNs) to model the dependencies between pixels. PixelRNN generates images pixel by pixel in a sequential manner, and it captures the spatial dependencies of pixels more effectively than traditional CNN-based methods.

In this project, we’ll implement a PixelRNN for image generation and show how it generates images sequentially.

About:
✅ What It Does:
Defines a PixelRNN model that generates images sequentially, pixel by pixel, using an LSTM network

Trains on the CIFAR-10 dataset to learn how to generate images one pixel at a time, based on the previously generated pixels

Uses MSE loss to minimize the difference between generated and real images

Generates 32x32 RGB images from random noise
"""


import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
 
# 1. Define the PixelRNN model
class PixelRNN(nn.Module):
    def __init__(self, in_channels=3, hidden_size=128, num_layers=2):
        super(PixelRNN, self).__init__()
 
        self.hidden_size = hidden_size
        self.num_layers = num_layers
 
        # Recurrent layers (LSTM)
        self.lstm = nn.LSTM(input_size=28 * 28, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
 
        # Output layer (for each pixel)
        self.fc = nn.Linear(hidden_size, 28 * 28 * in_channels)
 
    def forward(self, x):
        # Flatten the input image (28x28) to a vector
        x = x.view(-1, 28 * 28)  # Flatten the image to (batch_size, 28*28)
 
        # Pass through LSTM
        lstm_out, _ = self.lstm(x.unsqueeze(1))  # Add dummy sequence length dimension (1)
        
        # Output layer
        out = self.fc(lstm_out)
        return out.view(-1, 3, 28, 28)  # Reshape back to the image shape (3x28x28 for RGB)
 
# 2. Define the loss function and optimizer
criterion = nn.MSELoss()  # Using MSE loss for image generation tasks
optimizer = optim.Adam(model.parameters(), lr=0.0002)
 
# 3. Load the CIFAR-10 dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
 
# 4. Training loop for PixelRNN
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = PixelRNN().to(device)
 
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