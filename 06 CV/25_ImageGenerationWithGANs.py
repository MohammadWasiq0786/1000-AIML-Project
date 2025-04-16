"""
Project 225. Image generation with GANs
Description:
Image generation with GANs (Generative Adversarial Networks) involves training a model to create realistic images from random noise. A GAN consists of two networks: a generator (creates images) and a discriminator (judges real vs. fake). In this project, we’ll implement a simple GAN using PyTorch to generate handwritten digits (MNIST dataset) from scratch.

About:
What It Does:
This project demonstrates how to train a simple GAN to generate realistic-looking digits from noise. It's foundational for tasks like synthetic data generation, art creation, super-resolution, face generation, and much more. You can extend it using DCGAN, StyleGAN, or conditional GANs for more powerful image generation.
"""
# Install required packages:
# pip install torch torchvision matplotlib
 
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
 
# Hyperparameters
latent_dim = 100
batch_size = 128
image_size = 28 * 28
num_epochs = 30
lr = 0.0002
 
# Transform: convert images to tensor and normalize to [-1, 1]
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])
 
# Load MNIST dataset
dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
 
# Generator network: takes random noise and generates a 28x28 image
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.gen = nn.Sequential(
            nn.Linear(latent_dim, 256), nn.ReLU(True),
            nn.Linear(256, 512), nn.ReLU(True),
            nn.Linear(512, 1024), nn.ReLU(True),
            nn.Linear(1024, image_size), nn.Tanh()  # Output values in [-1, 1]
        )
 
    def forward(self, z):
        return self.gen(z)
 
# Discriminator network: takes an image and predicts if it's real or fake
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.dis = nn.Sequential(
            nn.Linear(image_size, 512), nn.LeakyReLU(0.2),
            nn.Linear(512, 256), nn.LeakyReLU(0.2),
            nn.Linear(256, 1), nn.Sigmoid()
        )
 
    def forward(self, x):
        return self.dis(x)
 
# Initialize models
generator = Generator()
discriminator = Discriminator()
 
# Loss function and optimizers
criterion = nn.BCELoss()
optim_G = optim.Adam(generator.parameters(), lr=lr)
optim_D = optim.Adam(discriminator.parameters(), lr=lr)
 
# Training loop
for epoch in range(num_epochs):
    for i, (imgs, _) in enumerate(loader):
        real_imgs = imgs.view(-1, image_size)
        batch_size = real_imgs.size(0)
 
        # Labels: real = 1, fake = 0
        real_labels = torch.ones(batch_size, 1)
        fake_labels = torch.zeros(batch_size, 1)
 
        # --- Train Discriminator ---
        z = torch.randn(batch_size, latent_dim)
        fake_imgs = generator(z)
        real_loss = criterion(discriminator(real_imgs), real_labels)
        fake_loss = criterion(discriminator(fake_imgs.detach()), fake_labels)
        d_loss = real_loss + fake_loss
 
        discriminator.zero_grad()
        d_loss.backward()
        optim_D.step()
 
        # --- Train Generator ---
        z = torch.randn(batch_size, latent_dim)
        generated = generator(z)
        g_loss = criterion(discriminator(generated), real_labels)
 
        generator.zero_grad()
        g_loss.backward()
        optim_G.step()
 
    print(f"Epoch [{epoch+1}/{num_epochs}]  Loss_D: {d_loss.item():.4f}  Loss_G: {g_loss.item():.4f}")
 
# Generate and display sample images
z = torch.randn(16, latent_dim)
sample_imgs = generator(z).view(-1, 1, 28, 28)
 
# Plot generated images
grid = torch.cat([img.squeeze(0) for img in sample_imgs], dim=1)
plt.figure(figsize=(12, 2))
plt.imshow(grid.detach().numpy(), cmap='gray')
plt.axis('off')
plt.title("Generated Digits with GAN")
plt.show()