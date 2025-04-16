"""
Project 378. Video generation models
Description:
Video generation models are designed to create sequences of frames that represent a continuous video. These models are particularly challenging due to the need to account for both spatial (image) and temporal (motion) dependencies between frames. Popular techniques used in video generation include GANs, VAEs, and RNNs. In this project, we'll explore a basic approach to video generation using GANs to generate a sequence of frames.

About:
✅ What It Does:
Generator creates a video by generating a sequence of frames from random noise (z_dim is the latent space size).

Discriminator evaluates if the generated video is real or fake.

Training involves a standard GAN approach where the Discriminator is trained to distinguish real from fake videos, and the Generator is trained to produce videos that the Discriminator classifies as real.

Generates 64x64 RGB videos with 10 frames per video.
"""


import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
 
# 1. Define the Generator model for Video Generation
class VideoGenerator(nn.Module):
    def __init__(self, z_dim=100, num_frames=10, img_channels=3, img_size=64):
        super(VideoGenerator, self).__init__()
 
        self.fc1 = nn.Linear(z_dim, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, 1024)
        self.fc4 = nn.Linear(1024, img_channels * num_frames * img_size * img_size)  # Output for video frames
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
 
    def forward(self, z):
        x = self.relu(self.fc1(z))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return self.tanh(x).view(-1, 3, 10, 64, 64)  # Reshape to video: 10 frames, 64x64 images
 
# 2. Define the Discriminator model for Video Generation
class VideoDiscriminator(nn.Module):
    def __init__(self, img_channels=3, img_size=64, num_frames=10):
        super(VideoDiscriminator, self).__init__()
 
        self.conv1 = nn.Conv3d(img_channels, 64, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv3d(64, 128, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv3d(128, 256, kernel_size=4, stride=2, padding=1)
        self.fc1 = nn.Linear(256 * (img_size // 8) * (img_size // 8) * num_frames, 1024)
        self.fc2 = nn.Linear(1024, 1)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.sigmoid = nn.Sigmoid()
 
    def forward(self, x):
        x = self.leaky_relu(self.conv1(x))
        x = self.leaky_relu(self.conv2(x))
        x = self.leaky_relu(self.conv3(x))
        x = x.view(x.size(0), -1)  # Flatten
        x = self.leaky_relu(self.fc1(x))
        return self.sigmoid(self.fc2(x))  # Output a probability
 
# 3. Initialize the models and define loss function and optimizers
z_dim = 100  # Latent vector dimension
num_frames = 10  # Number of frames in the video
img_size = 64  # Image size (64x64)
generator = VideoGenerator(z_dim=z_dim, num_frames=num_frames, img_size=img_size)
discriminator = VideoDiscriminator(img_size=img_size, num_frames=num_frames)
 
criterion = nn.BCELoss()
optimizer_g = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_d = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
 
# 4. Training loop for Video Generation Model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
generator.to(device)
discriminator.to(device)
 
num_epochs = 50
for epoch in range(num_epochs):
    for i in range(100):  # Assume we have 100 batches per epoch (for simplicity)
        real_video = torch.randn(64, 3, 10, 64, 64).to(device)  # Fake real videos for simplicity
        
        # Create labels for real and fake data
        real_labels = torch.ones(real_video.size(0), 1).to(device)
        fake_labels = torch.zeros(real_video.size(0), 1).to(device)
 
        # Train the Discriminator: Maximize log(D(x)) + log(1 - D(G(z)))
        optimizer_d.zero_grad()
 
        real_outputs = discriminator(real_video)
        d_loss_real = criterion(real_outputs, real_labels)
 
        z = torch.randn(real_video.size(0), z_dim).to(device)
        fake_video = generator(z)
        fake_outputs = discriminator(fake_video.detach())
        d_loss_fake = criterion(fake_outputs, fake_labels)
 
        d_loss = d_loss_real + d_loss_fake
        d_loss.backward()
        optimizer_d.step()
 
        # Train the Generator: Maximize log(1 - D(G(z))) = Minimize log(D(G(z)))
        optimizer_g.zero_grad()
 
        fake_outputs = discriminator(fake_video)
        g_loss = criterion(fake_outputs, real_labels)
        g_loss.backward()
        optimizer_g.step()
 
    # Print loss every few epochs
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}')
 
    # Generate and display sample videos every few epochs
    if (epoch + 1) % 10 == 0:
        with torch.no_grad():
            z = torch.randn(64, z_dim).to(device)
            fake_video = generator(z).cpu()
            fake_video = fake_video.squeeze(0).detach().numpy()
            plt.imshow(fake_video[0].transpose(1, 2, 0))  # Display the first frame
            plt.title(f"Generated Video at Epoch {epoch + 1}")
            plt.show()