"""
Project 385. Data augmentation with generative models
Description:
Data augmentation is a technique used in machine learning to artificially increase the size of a dataset by creating modified versions of existing data. Generative models like GANs and VAEs can be used for data augmentation by generating new, synthetic data that resembles the original dataset. This is particularly useful when working with limited data, and it helps improve the performance of models by introducing more variation.

In this project, we will use a GAN for image data augmentation, where the model generates new images that resemble the original dataset.

✅ What It Does:
Generator generates synthetic images from random noise, which are intended to augment the existing dataset.

Discriminator distinguishes between real and fake images, enabling the Generator to improve its output.

Data Augmentation allows the model to generate new MNIST-like images, increasing the diversity of the dataset.

The model can be trained for image data augmentation and can be extended for more complex datasets.
"""



import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
 
# 1. Define the Generator model for Data Augmentation
class Generator(nn.Module):
    def __init__(self, z_dim=100):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(z_dim, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, 1024)
        self.fc4 = nn.Linear(1024, 28 * 28)  # Output 28x28 image (MNIST)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
 
    def forward(self, z):
        x = self.relu(self.fc1(z))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return self.tanh(x).view(-1, 1, 28, 28)  # Reshape to image
 
# 2. Define the Discriminator model for Data Augmentation
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.fc1 = nn.Linear(128 * 7 * 7, 1024)
        self.fc2 = nn.Linear(1024, 1)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.sigmoid = nn.Sigmoid()
 
    def forward(self, x):
        x = self.leaky_relu(self.conv1(x))
        x = self.leaky_relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # Flatten the image
        x = self.leaky_relu(self.fc1(x))
        return self.sigmoid(self.fc2(x))  # Output real/fake
 
# 3. Load the MNIST dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
 
# 4. Loss function and optimizer
criterion = nn.BCELoss()
optimizer_g = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_d = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
 
# 5. Train the model for data augmentation
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
generator = Generator(z_dim=100).to(device)
discriminator = Discriminator().to(device)
 
num_epochs = 50
for epoch in range(num_epochs):
    for i, (real_images, _) in enumerate(train_loader):
        real_images = real_images.to(device)
        
        # Create labels for real and fake data
        real_labels = torch.ones(real_images.size(0), 1).to(device)
        fake_labels = torch.zeros(real_images.size(0), 1).to(device)
 
        # Train the Discriminator: Maximize log(D(x)) + log(1 - D(G(z)))
        optimizer_d.zero_grad()
 
        real_outputs = discriminator(real_images)
        d_loss_real = criterion(real_outputs, real_labels)
 
        z = torch.randn(real_images.size(0), 100).to(device)  # Random noise
        fake_images = generator(z)
        fake_outputs = discriminator(fake_images.detach())
        d_loss_fake = criterion(fake_outputs, fake_labels)
 
        d_loss = d_loss_real + d_loss_fake
        d_loss.backward()
        optimizer_d.step()
 
        # Train the Generator: Maximize log(1 - D(G(z))) = Minimize log(D(G(z)))
        optimizer_g.zero_grad()
 
        fake_outputs = discriminator(fake_images)
        g_loss = criterion(fake_outputs, real_labels)
        g_loss.backward()
        optimizer_g.step()
 
    # Print loss every few epochs
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}')
 
    # Generate and display augmented images every few epochs
    if (epoch + 1) % 10 == 0:
        with torch.no_grad():
            z = torch.randn(64, 100).to(device)
            fake_images = generator(z).cpu()
            grid_img = torchvision.utils.make_grid(fake_images, nrow=8, normalize=True)
            plt.imshow(grid_img.permute(1, 2, 0))
            plt.title(f"Augmented Images at Epoch {epoch + 1}")
            plt.show()