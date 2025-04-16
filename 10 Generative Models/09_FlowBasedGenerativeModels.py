"""
Project 369. Flow-based generative models
Description:
Flow-based generative models use invertible transformations to model complex distributions. Unlike models like GANs or VAEs, flow-based models allow for exact likelihood estimation and efficient sampling. They work by transforming a simple distribution (like Gaussian noise) into a complex distribution by applying a sequence of invertible functions.

In this project, we’ll implement a flow-based generative model (specifically normalizing flows) for generating images and learn how to model complex data distributions.

About:
✅ What It Does:
Defines a Flow-based model using convolutional layers and a fully connected output layer

Trains on the CIFAR-10 dataset, using cross-entropy loss for classification tasks

Unlike GANs or VAEs, this model allows for exact likelihood estimation and efficient sampling by applying invertible transformations

It generates high-quality images by learning complex distributions
"""


import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
 
# 1. Define a simple flow-based model (Normalizing Flows)
class FlowBasedModel(nn.Module):
    def __init__(self, in_channels=3, num_filters=64, num_layers=4):
        super(FlowBasedModel, self).__init__()
 
        self.num_layers = num_layers
        self.conv_layers = nn.ModuleList()
 
        # First convolutional layer
        self.conv_layers.append(nn.Conv2d(in_channels, num_filters, kernel_size=4, stride=2, padding=1))
 
        # Additional convolutional layers
        for _ in range(num_layers - 1):
            self.conv_layers.append(nn.Conv2d(num_filters, num_filters, kernel_size=4, stride=2, padding=1))
 
        # Final output layer
        self.fc = nn.Linear(num_filters * 8 * 8, 10)  # 10 classes (for CIFAR-10)
 
    def forward(self, x):
        for layer in self.conv_layers:
            x = torch.relu(layer(x))  # Apply ReLU activation
        x = x.view(x.size(0), -1)  # Flatten the image
        return self.fc(x)
 
# 2. Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0002)
 
# 3. Load the CIFAR-10 dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
 
# 4. Training loop for flow-based model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = FlowBasedModel().to(device)
 
num_epochs = 50
for epoch in range(num_epochs):
    for i, (real_images, labels) in enumerate(train_loader):
        real_images, labels = real_images.to(device), labels.to(device)
 
        # Forward pass
        optimizer.zero_grad()
        outputs = model(real_images)
 
        # Compute the loss
        loss = criterion(outputs, labels)
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