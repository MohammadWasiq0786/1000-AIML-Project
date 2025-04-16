"""
Project 364. CycleGAN for unpaired translation
Description:
A CycleGAN (Cycle-Consistent Generative Adversarial Network) is a type of GAN used for image-to-image translation when there are no paired images (i.e., images in one domain don't have corresponding images in another domain). This method enables tasks like:

Photo to painting conversion

Season transfer (e.g., summer to winter)

Image style transfer (e.g., turning a photo into a pencil sketch)

CycleGAN works by learning to generate images in one domain that are cycle-consistent with their counterparts in the other domain, without needing paired datasets.

In this project, we’ll implement a CycleGAN for unpaired image-to-image translation.

About:
✅ What It Does:
Generator A to B and Generator B to A generate images by learning to translate between domains (e.g., domain A to B and vice versa)

Discriminators are trained to distinguish between real and generated images in each domain

The Cycle Consistency Loss ensures that the translated image can be transformed back to the original domain, maintaining consistency across the translation

The Adversarial Loss pushes both the generator and discriminator to improve, generating realistic images that the discriminator cannot easily distinguish from real images
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch import nn
import matplotlib.pyplot as plt
 
# 1. Define the Generator Model (for CycleGAN)
class Generator(nn.Module):
    def __init__(self, in_channels, out_channels, num_filters=64):
        super(Generator, self).__init__()
 
        # Encoder layers
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, num_filters, kernel_size=4, stride=2, padding=1),  # Downsample
            nn.ReLU(inplace=True),
            nn.Conv2d(num_filters, num_filters*2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_filters*2, num_filters*4, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
 
        # Decoder layers
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(num_filters*4, num_filters*2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(num_filters*2, num_filters, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(num_filters, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),  # Output in the range [-1, 1]
        )
 
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
 
# 2. Define the Discriminator Model (for CycleGAN)
class Discriminator(nn.Module):
    def __init__(self, in_channels, num_filters=64):
        super(Discriminator, self).__init__()
 
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, num_filters, kernel_size=4, stride=2, padding=1),  # Downsample
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(num_filters, num_filters*2, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(num_filters*2, num_filters*4, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(num_filters*4, 1, kernel_size=4, stride=2, padding=1),  # Output a single value
            nn.Sigmoid()  # Output probability
        )
 
    def forward(self, x):
        return self.model(x)
 
# 3. Loss Functions for CycleGAN
def cycle_loss(real_img, fake_img, lambda_cycle=10.0):
    """Cycle Consistency Loss"""
    return lambda_cycle * torch.mean(torch.abs(real_img - fake_img))
 
def adversarial_loss(real_pred, fake_pred):
    """Adversarial loss for both generator and discriminator"""
    return torch.mean((real_pred - 1)**2 + fake_pred**2)
 
# 4. Initialize models, optimizers, and datasets
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
generator_A_to_B = Generator(3, 3).to(device)  # A to B translation
generator_B_to_A = Generator(3, 3).to(device)  # B to A translation
discriminator_A = Discriminator(3).to(device)
discriminator_B = Discriminator(3).to(device)
 
# Optimizers
lr = 0.0002
beta1 = 0.5
optimizer_g = optim.Adam(list(generator_A_to_B.parameters()) + list(generator_B_to_A.parameters()), lr=lr, betas=(beta1, 0.999))
optimizer_d = optim.Adam(list(discriminator_A.parameters()) + list(discriminator_B.parameters()), lr=lr, betas=(beta1, 0.999))
 
# 5. Training loop for CycleGAN
num_epochs = 50
for epoch in range(num_epochs):
    for i, (real_A, real_B) in enumerate(train_loader):  # train_loader should load unpaired datasets
        batch_size = real_A.size(0)
        real_A, real_B = real_A.to(device), real_B.to(device)
 
        # Labels for real and fake data
        real_labels = torch.ones(batch_size, 1).to(device)
        fake_labels = torch.zeros(batch_size, 1).to(device)
 
        # Train Discriminators
        optimizer_d.zero_grad()
 
        # Discriminator A (real vs fake)
        real_A_pred = discriminator_A(real_A)
        fake_A = generator_B_to_A(real_B).detach()
        fake_A_pred = discriminator_A(fake_A)
        d_loss_A = adversarial_loss(real_A_pred, fake_A_pred)
 
        # Discriminator B (real vs fake)
        real_B_pred = discriminator_B(real_B)
        fake_B = generator_A_to_B(real_A).detach()
        fake_B_pred = discriminator_B(fake_B)
        d_loss_B = adversarial_loss(real_B_pred, fake_B_pred)
 
        d_loss = d_loss_A + d_loss_B
        d_loss.backward()
        optimizer_d.step()
 
        # Train Generators
        optimizer_g.zero_grad()
 
        # Generator A to B
        fake_B = generator_A_to_B(real_A)
        real_B_pred = discriminator_B(fake_B)
        g_loss_A_to_B = adversarial_loss(real_B_pred, real_labels)
 
        # Generator B to A
        fake_A = generator_B_to_A(real_B)
        real_A_pred = discriminator_A(fake_A)
        g_loss_B_to_A = adversarial_loss(real_A_pred, real_labels)
 
        # Cycle Consistency Loss
        rec_A = generator_B_to_A(fake_B)
        rec_B = generator_A_to_B(fake_A)
        cycle_loss_A = cycle_loss(real_A, rec_A)
        cycle_loss_B = cycle_loss(real_B, rec_B)
 
        g_loss = g_loss_A_to_B + g_loss_B_to_A + cycle_loss_A + cycle_loss_B
        g_loss.backward()
        optimizer_g.step()
 
    # Print loss every few epochs
    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}')
 
    # Generate and save sample images
    if (epoch+1) % 10 == 0:
        with torch.no_grad():
            fake_images_A_to_B = generator_A_to_B(real_A).detach().cpu()
            grid_img = torchvision.utils.make_grid(fake_images_A_to_B, nrow=8, normalize=True)
            plt.imshow(grid_img.permute(1, 2, 0))
            plt.show()