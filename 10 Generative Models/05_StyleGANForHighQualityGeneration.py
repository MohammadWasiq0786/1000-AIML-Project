"""
Project 365. StyleGAN for high-quality generation
Description:
StyleGAN (Style Generative Adversarial Network) is a state-of-the-art generative model known for generating high-quality, photorealistic images. StyleGAN introduces a style-based generator architecture that allows for fine-grained control over the generated images, making it possible to manipulate features like texture, color, and overall style at various layers of the network.

In this project, we’ll implement StyleGAN for generating high-resolution images (such as faces or other objects) with fine control over the generated styles.

About:
✅ What It Does:
Defines a StyleGAN Generator and Discriminator for high-quality image generation

Uses adversarial loss to train the generator to create realistic images and the discriminator to differentiate between real and fake images

The generator creates images based on random noise (latent vector) and the discriminator classifies them as real or fake

Generates 64x64 images (using CIFAR-10 dataset) through adversarial training
"""


import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
 
# 1. Define the StyleGAN Generator Architecture
class StyleGANGenerator(nn.Module):
    def __init__(self, z_dim=512, c_dim=512):
        super(StyleGANGenerator, self).__init__()
        
        # StyleGAN layers are often composed of multiple transposed convolutions (deconvolutions)
        self.fc1 = nn.Linear(z_dim, c_dim)
        self.fc2 = nn.Linear(c_dim, c_dim*2)
        self.fc3 = nn.Linear(c_dim*2, c_dim*4)
        self.fc4 = nn.Linear(c_dim*4, c_dim*8)
        self.fc5 = nn.Linear(c_dim*8, 3 * 64 * 64)  # Output 64x64 image
        
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()  # For output range [-1, 1]
    
    def forward(self, z):
        z = self.relu(self.fc1(z))
        z = self.relu(self.fc2(z))
        z = self.relu(self.fc3(z))
        z = self.relu(self.fc4(z))
        z = self.fc5(z)
        return self.tanh(z).view(-1, 3, 64, 64)  # Reshape to 64x64 image (RGB)
 
# 2. Define the Discriminator for StyleGAN
class StyleGANDiscriminator(nn.Module):
    def __init__(self):
        super(StyleGANDiscriminator, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1)
        self.fc = nn.Linear(512 * 4 * 4, 1)  # Flattened before final output layer
        
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.sigmoid = nn.Sigmoid()  # Output probability of real/fake
    
    def forward(self, x):
        x = self.leaky_relu(self.conv1(x))
        x = self.leaky_relu(self.conv2(x))
        x = self.leaky_relu(self.conv3(x))
        x = self.leaky_relu(self.conv4(x))
        x = x.view(x.size(0), -1)  # Flatten
        return self.sigmoid(self.fc(x))
 
# 3. Initialize models, loss functions, and optimizers
z_dim = 512  # Latent vector dimension for StyleGAN
generator = StyleGANGenerator(z_dim=z_dim)
discriminator = StyleGANDiscriminator()
 
# Binary cross-entropy loss for adversarial training
criterion = nn.BCELoss()
 
# Optimizers for both generator and discriminator
lr = 0.0002
beta1 = 0.5
optimizer_g = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, 0.999))
optimizer_d = optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, 0.999))
 
# 4. Load dataset (using a sample dataset for demonstration)
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
 
# 5. Training loop for StyleGAN
num_epochs = 50
for epoch in range(num_epochs):
    for i, (real_images, _) in enumerate(train_loader):
        batch_size = real_images.size(0)
        real_images = real_images.cuda()
        
        # Create labels for real and fake data
        real_labels = torch.ones(batch_size, 1).cuda()  # Real data labels are 1
        fake_labels = torch.zeros(batch_size, 1).cuda()  # Fake data labels are 0
 
        # Train the Discriminator: Maximize log(D(x)) + log(1 - D(G(z)))
        optimizer_d.zero_grad()
 
        # Train on real images
        real_pred = discriminator(real_images)
        d_loss_real = criterion(real_pred, real_labels)
        d_loss_real.backward()
 
        # Train on fake images
        z = torch.randn(batch_size, z_dim).cuda()  # Random noise for generator input
        fake_images = generator(z)
        fake_pred = discriminator(fake_images.detach())  # Detach to avoid training generator
        d_loss_fake = criterion(fake_pred, fake_labels)
        d_loss_fake.backward()
        
        # Update the discriminator
        d_loss = d_loss_real + d_loss_fake
        optimizer_d.step()
        
        # Train the Generator: Minimize log(1 - D(G(z))) = Maximize log(D(G(z)))
        optimizer_g.zero_grad()
        
        fake_pred = discriminator(fake_images)
        g_loss = criterion(fake_pred, real_labels)  # We want fake images to be classified as real
        g_loss.backward()
        
        # Update the generator
        optimizer_g.step()
 
    # Print loss every few epochs
    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}')
    
    # Generate and display sample images
    if (epoch+1) % 10 == 0:
        with torch.no_grad():
            fake_images = generator(torch.randn(64, z_dim).cuda()).cpu()
            grid_img = torchvision.utils.make_grid(fake_images, nrow=8, normalize=True)
            plt.imshow(grid_img.permute(1, 2, 0))
            plt.show()