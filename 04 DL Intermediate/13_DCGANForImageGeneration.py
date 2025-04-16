"""
Project 133. DCGAN for image generation
Description:
DCGAN (Deep Convolutional Generative Adversarial Network) is a GAN architecture that uses convolutional layers to generate realistic images from random noise. In this project, we implement a DCGAN using PyTorch to generate images (e.g., digits, faces) by training a generator and discriminator adversarially.

About:
🧠 What This Project Demonstrates:
Implements a DCGAN architecture with transpose convolutions

Trains a generator and discriminator adversarially

Generates realistic grayscale images from random noise
"""

# Install if not already: pip install torch torchvision matplotlib
 
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
 
# Hyperparameters
latent_dim = 100
image_size = 64
batch_size = 128
num_epochs = 1  # For quick demo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 
# Transform and load dataset
transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])  # Normalize to [-1, 1]
])
 
dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
 
# Define Generator
class Generator(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 256, 4, 1, 0, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(True),
 
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(True),
 
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(True),
 
            nn.ConvTranspose2d(64, 1, 4, 2, 1, bias=False),
            nn.Tanh()
        )
 
    def forward(self, z):
        return self.main(z)
 
# Define Discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(1, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
 
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128), nn.LeakyReLU(0.2, inplace=True),
 
            nn.Conv2d(128, 1, 7, 1, 0, bias=False),
            nn.Sigmoid()
        )
 
    def forward(self, img):
        return self.main(img).view(-1, 1).squeeze(1)
 
# Initialize models
G = Generator(latent_dim).to(device)
D = Discriminator().to(device)
 
# Loss and optimizers
criterion = nn.BCELoss()
optimizer_G = torch.optim.Adam(G.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_D = torch.optim.Adam(D.parameters(), lr=0.0002, betas=(0.5, 0.999))
 
# Training loop (1 epoch for demo)
for epoch in range(num_epochs):
    for i, (real_imgs, _) in enumerate(dataloader):
        real_imgs = real_imgs.to(device)
        batch_size = real_imgs.size(0)
 
        # Real and fake labels
        valid = torch.ones(batch_size, device=device)
        fake = torch.zeros(batch_size, device=device)
 
        # Train Discriminator
        z = torch.randn(batch_size, latent_dim, 1, 1, device=device)
        fake_imgs = G(z).detach()
        loss_D = criterion(D(real_imgs), valid) + criterion(D(fake_imgs), fake)
        optimizer_D.zero_grad()
        loss_D.backward()
        optimizer_D.step()
 
        # Train Generator
        z = torch.randn(batch_size, latent_dim, 1, 1, device=device)
        gen_imgs = G(z)
        loss_G = criterion(D(gen_imgs), valid)
        optimizer_G.zero_grad()
        loss_G.backward()
        optimizer_G.step()
 
        if i % 200 == 0:
            print(f"Epoch {epoch} Batch {i} | Loss D: {loss_D:.4f}, Loss G: {loss_G:.4f}")
 
# Generate and show a few images
G.eval()
with torch.no_grad():
    z = torch.randn(16, latent_dim, 1, 1, device=device)
    gen_imgs = G(z).cpu()
 
grid = torchvision.utils.make_grid(gen_imgs, nrow=4, normalize=True)
plt.figure(figsize=(6, 6))
plt.imshow(grid.permute(1, 2, 0))
plt.title("🎨 DCGAN Generated Images")
plt.axis("off")
plt.show()