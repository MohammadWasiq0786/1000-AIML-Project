"""
Project 362. Generative adversarial networks
Description:
Generative Adversarial Networks (GANs) consist of two neural networks: a generator and a discriminator, which compete against each other. The generator creates fake data (e.g., images), and the discriminator tries to distinguish between real and fake data. The goal is for the generator to create realistic data that the discriminator cannot easily distinguish from real data.

GANs are used for tasks like image generation, style transfer, and super-resolution.

In this project, we’ll implement a basic GAN to generate images from random noise and train it using the MNIST dataset.

About:
✅ What It Does:
Defines Generator and Discriminator models as part of the GAN architecture

Trains the discriminator to distinguish between real and fake images and the generator to create convincing fake images

Uses binary cross-entropy loss for both the generator and discriminator

Trains on the MNIST dataset, generating new images similar to handwritten digits
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
 
# 1. Define the Generator model
class Generator(nn.Module):
    def __init__(self, z_dim=100):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(z_dim, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, 1024)
        self.fc4 = nn.Linear(1024, 28 * 28)  # 28x28 image size (MNIST)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()  # For output to be in range [-1, 1]
 
    def forward(self, z):
        z = self.relu(self.fc1(z))
        z = self.relu(self.fc2(z))
        z = self.relu(self.fc3(z))
        z = self.tanh(self.fc4(z))
        return z.view(-1, 1, 28, 28)  # Reshape to image dimensions
 
# 2. Define the Discriminator model
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 1)  # Output single value: real or fake
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.sigmoid = nn.Sigmoid()
 
    def forward(self, x):
        x = x.view(-1, 28 * 28)  # Flatten the image
        x = self.leaky_relu(self.fc1(x))
        x = self.leaky_relu(self.fc2(x))
        x = self.leaky_relu(self.fc3(x))
        x = self.sigmoid(self.fc4(x))
        return x
 
# 3. Initialize models, loss function, and optimizers
z_dim = 100  # Latent vector dimension (noise)
generator = Generator(z_dim)
discriminator = Discriminator()
 
# Binary cross-entropy loss
criterion = nn.BCELoss()
 
# Optimizers for both generator and discriminator
lr = 0.0002
beta1 = 0.5
optimizer_g = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, 0.999))
optimizer_d = optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, 0.999))
 
# 4. Load the MNIST dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
 
# 5. Training loop for GAN
num_epochs = 50
for epoch in range(num_epochs):
    for i, (real_images, _) in enumerate(train_loader):
        batch_size = real_images.size(0)
        
        # Create labels for real and fake data
        real_labels = torch.ones(batch_size, 1)  # Real data labels are 1
        fake_labels = torch.zeros(batch_size, 1)  # Fake data labels are 0
        
        # Train the Discriminator: Maximize log(D(x)) + log(1 - D(G(z)))
        optimizer_d.zero_grad()
        
        # Train on real images
        outputs = discriminator(real_images)
        d_loss_real = criterion(outputs, real_labels)
        d_loss_real.backward()
 
        # Train on fake images
        z = torch.randn(batch_size, z_dim)  # Random noise for generator input
        fake_images = generator(z)
        outputs = discriminator(fake_images.detach())  # Detach to avoid training generator
        d_loss_fake = criterion(outputs, fake_labels)
        d_loss_fake.backward()
        
        # Update the discriminator
        d_loss = d_loss_real + d_loss_fake
        optimizer_d.step()
        
        # Train the Generator: Minimize log(1 - D(G(z))) = Maximize log(D(G(z)))
        optimizer_g.zero_grad()
        
        # Try to fool the discriminator with fake images
        outputs = discriminator(fake_images)
        g_loss = criterion(outputs, real_labels)  # We want fake images to be classified as real
        g_loss.backward()
        
        # Update the generator
        optimizer_g.step()
 
    # Print loss every 10 epochs
    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], d_loss: {d_loss.item():.4f}, g_loss: {g_loss.item():.4f}')
    
    # Generate and save sample images
    if (epoch+1) % 10 == 0:
        with torch.no_grad():
            fake_images = generator(torch.randn(64, z_dim)).detach().cpu()
            grid_img = torchvision.utils.make_grid(fake_images, nrow=8, normalize=True)
            plt.imshow(grid_img.permute(1, 2, 0))
            plt.show()