"""
Project 363. Conditional GAN implementation
Description:
A Conditional Generative Adversarial Network (cGAN) is an extension of the GAN architecture that generates images based on conditional information, such as labels. For example, a cGAN can generate images of handwritten digits (from MNIST) based on a digit label (0-9), or it can generate images of clothes conditioned on the type of clothing (e.g., shirts, pants).

In this project, we’ll implement a cGAN where the generator and discriminator take both random noise and a label as input, allowing for the generation of specific types of images based on the provided label.

About:
✅ What It Does:
Defines a Conditional GAN where both the generator and discriminator take a label (condition) along with the usual inputs

Trains the discriminator to distinguish between real and fake images and the generator to create images conditioned on a label

Uses one-hot encoded labels to condition the generation of specific images (e.g., generating a "0" or "1" from the MNIST dataset)

Generates MNIST digit images based on labels
"""


import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
 
# 1. Define the Generator model for cGAN
class Generator(nn.Module):
    def __init__(self, z_dim=100, label_dim=10):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(z_dim + label_dim, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, 1024)
        self.fc4 = nn.Linear(1024, 28 * 28)  # 28x28 image size (MNIST)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()  # For output to be in range [-1, 1]
 
    def forward(self, z, label):
        # Concatenate noise vector and label
        input = torch.cat((z, label), -1)
        x = self.relu(self.fc1(input))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.tanh(self.fc4(x))
        return x.view(-1, 1, 28, 28)  # Reshape to image dimensions
 
# 2. Define the Discriminator model for cGAN
class Discriminator(nn.Module):
    def __init__(self, label_dim=10):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(28 * 28 + label_dim, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 1)  # Output single value: real or fake
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.sigmoid = nn.Sigmoid()
 
    def forward(self, x, label):
        x = x.view(-1, 28 * 28)  # Flatten the image
        input = torch.cat((x, label), -1)  # Concatenate image and label
        x = self.leaky_relu(self.fc1(input))
        x = self.leaky_relu(self.fc2(x))
        x = self.leaky_relu(self.fc3(x))
        x = self.sigmoid(self.fc4(x))
        return x
 
# 3. Initialize models, loss function, and optimizers
z_dim = 100  # Latent vector dimension (noise)
label_dim = 10  # Number of classes for MNIST (0-9)
generator = Generator(z_dim, label_dim)
discriminator = Discriminator(label_dim)
 
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
 
# 5. Training loop for cGAN
num_epochs = 50
for epoch in range(num_epochs):
    for i, (real_images, labels) in enumerate(train_loader):
        batch_size = real_images.size(0)
        
        # Create labels for real and fake data
        real_labels = torch.ones(batch_size, 1)  # Real data labels are 1
        fake_labels = torch.zeros(batch_size, 1)  # Fake data labels are 0
        
        # One-hot encode labels
        label_one_hot = torch.zeros(batch_size, label_dim)
        label_one_hot.scatter_(1, labels.view(-1, 1), 1)
        
        # Train the Discriminator: Maximize log(D(x)) + log(1 - D(G(z)))
        optimizer_d.zero_grad()
        
        # Train on real images
        outputs = discriminator(real_images, label_one_hot)
        d_loss_real = criterion(outputs, real_labels)
        d_loss_real.backward()
 
        # Train on fake images
        z = torch.randn(batch_size, z_dim)  # Random noise for generator input
        fake_images = generator(z, label_one_hot)
        outputs = discriminator(fake_images.detach(), label_one_hot)  # Detach to avoid training generator
        d_loss_fake = criterion(outputs, fake_labels)
        d_loss_fake.backward()
        
        # Update the discriminator
        d_loss = d_loss_real + d_loss_fake
        optimizer_d.step()
        
        # Train the Generator: Minimize log(1 - D(G(z))) = Maximize log(D(G(z)))
        optimizer_g.zero_grad()
        
        # Try to fool the discriminator with fake images
        outputs = discriminator(fake_images, label_one_hot)
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
            fake_images = generator(torch.randn(64, z_dim), label_one_hot).detach().cpu()
            grid_img = torchvision.utils.make_grid(fake_images, nrow=8, normalize=True)
            plt.imshow(grid_img.permute(1, 2, 0))
            plt.show()