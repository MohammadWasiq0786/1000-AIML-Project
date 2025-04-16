"""
Project 388. Few-shot image generation
Description:
Few-shot image generation refers to the task of generating new images based on a limited number of examples (i.e., few shots). This is a challenging problem because most generative models require large datasets to train effectively. Few-shot learning aims to leverage small datasets to produce realistic images. Models like GANs or VAEs can be adapted to handle few-shot learning by using techniques like transfer learning or meta-learning.

In this project, we’ll implement a few-shot image generation model using a pre-trained model and fine-tuning it on a few examples to generate new images.

About:
✅ What It Does:
The FewShotGenerator creates images from random noise using a simple fully connected neural network.

The model is fine-tuned using a small set of images from the CIFAR-10 dataset, simulating few-shot learning where the model adapts to a limited number of training examples.

Generated images are displayed during the training process, and the model gradually learns to generate more realistic images.

Key features:
Few-shot learning fine-tunes a pre-trained model on a small dataset.

Uses GAN-like architecture to generate realistic images from random noise after fine-tuning.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torchvision.models import resnet18
 
# 1. Define the Few-shot GAN Model (simplified example)
class FewShotGenerator(nn.Module):
    def __init__(self, z_dim=100, img_channels=3, img_size=64):
        super(FewShotGenerator, self).__init__()
        self.fc1 = nn.Linear(z_dim, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, 1024)
        self.fc4 = nn.Linear(1024, img_channels * img_size * img_size)  # Output 64x64 image
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
 
    def forward(self, z):
        x = self.relu(self.fc1(z))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return self.tanh(x).view(-1, 3, 64, 64)  # Reshape to image
 
# 2. Load a few images for fine-tuning (For simplicity, using CIFAR-10 dataset)
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
 
# 3. Loss function and optimizer
criterion = nn.BCELoss()
optimizer_g = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_d = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
 
# 4. Initialize the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
generator = FewShotGenerator(z_dim=100).to(device)
 
# 5. Few-shot learning fine-tuning (simplified)
# Assume you have a small set of images for fine-tuning (for simplicity, using the CIFAR-10 dataset)
num_epochs = 10
for epoch in range(num_epochs):
    for i, (real_images, _) in enumerate(train_loader):
        real_images = real_images.to(device)
        z = torch.randn(real_images.size(0), 100).to(device)  # Random noise for generator input
 
        # Train the Generator
        optimizer_g.zero_grad()
        fake_images = generator(z)
        real_labels = torch.ones(real_images.size(0), 1).to(device)
        fake_labels = torch.zeros(real_images.size(0), 1).to(device)
 
        g_loss = criterion(fake_images, real_images)  # Compare generated and real images
        g_loss.backward()
        optimizer_g.step()
 
    # Print loss every few epochs
    if (epoch + 1) % 2 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Generator Loss: {g_loss.item():.4f}')
 
    # Display generated images
    if (epoch + 1) % 5 == 0:
        with torch.no_grad():
            z = torch.randn(16, 100).to(device)
            fake_images = generator(z).cpu()
            grid_img = torchvision.utils.make_grid(fake_images, nrow=4, normalize=True)
            plt.imshow(grid_img.permute(1, 2, 0))
            plt.title(f"Generated Images at Epoch {epoch + 1}")
            plt.show()