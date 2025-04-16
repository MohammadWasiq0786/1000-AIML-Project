"""
Project 480. Medical image generation
Description:
Medical image generation involves creating synthetic medical images, such as X-rays, CT scans, or MRI images, for training models, aiding diagnostics, or augmenting datasets. In this project, we simulate image generation using Generative Adversarial Networks (GANs) to create synthetic X-ray images.

About:
✅ What It Does:
Implements a basic GAN with a Generator and Discriminator for generating synthetic 28x28 images (simulated X-rays).

Trains the Discriminator to distinguish between real and fake images, and the Generator to create images that "fool" the Discriminator.

Can be extended to:

Use real medical images for training

Implement more complex architectures like DCGAN, Pix2Pix, or StyleGAN

Scale to larger medical images (e.g., CT scans, MRIs)

Real-world datasets:

NIH Chest X-ray Dataset, LUNA16, or LIDC-IDRI
"""

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
 
# 1. Define a simple GAN architecture
 
# Generator
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(100, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, 1024)
        self.fc4 = nn.Linear(1024, 28*28)  # Example size, can be adjusted for X-ray images
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
 
    def forward(self, z):
        x = self.relu(self.fc1(z))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.tanh(self.fc4(x))
        return x.view(-1, 1, 28, 28)  # Reshape to image format
 
# Discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(28*28, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 1)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.sigmoid = nn.Sigmoid()
 
    def forward(self, x):
        x = x.view(-1, 28*28)  # Flatten the image
        x = self.leaky_relu(self.fc1(x))
        x = self.leaky_relu(self.fc2(x))
        x = self.leaky_relu(self.fc3(x))
        x = self.sigmoid(self.fc4(x))
        return x
 
# 2. Set up GAN training
generator = Generator()
discriminator = Discriminator()
 
criterion = nn.BCELoss()
optimizer_G = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
 
# 3. Training loop
num_epochs = 100
for epoch in range(num_epochs):
    # Train Discriminator
    real_images = torch.randn(32, 1, 28, 28)  # Simulate real medical images
    labels_real = torch.ones(32, 1)
    labels_fake = torch.zeros(32, 1)
 
    optimizer_D.zero_grad()
    output_real = discriminator(real_images)
    loss_real = criterion(output_real, labels_real)
    z = torch.randn(32, 100)  # Latent space input
    fake_images = generator(z)
    output_fake = discriminator(fake_images.detach())
    loss_fake = criterion(output_fake, labels_fake)
    loss_D = loss_real + loss_fake
    loss_D.backward()
    optimizer_D.step()
 
    # Train Generator
    optimizer_G.zero_grad()
    output_fake = discriminator(fake_images)
    loss_G = criterion(output_fake, labels_real)  # We want to fool the discriminator
    loss_G.backward()
    optimizer_G.step()
 
    # Output training progress
    if epoch % 10 == 0:
        print(f"Epoch [{epoch}/{num_epochs}] | Loss D: {loss_D.item():.4f} | Loss G: {loss_G.item():.4f}")
 
    # Plot generated images
    if epoch % 20 == 0:
        plt.imshow(fake_images[0].detach().numpy().reshape(28, 28), cmap='gray')
        plt.title(f"Generated Image at Epoch {epoch}")
        plt.show()