"""
Project 373. Score-based generative models
Description:
Score-based generative models are a type of generative model that work by learning the gradient (score) of the data distribution with respect to data points. These models generate data by sampling from a simple distribution (e.g., Gaussian) and iteratively refining the samples using the learned score function. The primary advantage of score-based models is that they allow for direct sampling and can be trained without adversarial loss.

In this project, we’ll implement a Score-based Generative Model for generating images by using score matching techniques to refine random noise into data samples.

About:
✅ What It Does:
Defines a Score-based generative model that generates images from random noise by refining the noise through a learned score function

Uses Mean Squared Error (MSE) as a simple proxy for score matching, encouraging the generated images to match the real images

Trains on the CIFAR-10 dataset to generate 32x32 RGB images

Generates high-quality images by refining random noise using the learned gradients
"""


import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
 
# 1. Define the Score-based Model
class ScoreBasedModel(nn.Module):
    def __init__(self, in_channels=3, num_filters=64):
        super(ScoreBasedModel, self).__init__()
 
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
 
# 4. Score matching function (simplified version)
def score_matching_loss(real_images, generated_images):
    # Simple score matching loss using Mean Squared Error (MSE)
    return torch.mean((real_images - generated_images) ** 2)
 
# 5. Training loop for Score-based Model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ScoreBasedModel().to(device)
 
num_epochs = 50
for epoch in range(num_epochs):
    for i, (real_images, _) in enumerate(train_loader):
        real_images = real_images.to(device)
 
        # Generate random noise as input for score-based model
        noise = torch.randn(real_images.size()).to(device)
 
        # Forward pass
        optimizer.zero_grad()
        generated_images = model(noise)
 
        # Compute the score matching loss (refining generated images to match real images)
        loss = score_matching_loss(real_images, generated_images)
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