"""
Project 399. GAN training stabilization techniques
Description:
Training Generative Adversarial Networks (GANs) can be notoriously unstable, with issues such as mode collapse (where the generator produces the same output for all inputs) or gradient vanishing/exploding. Several training stabilization techniques have been proposed to mitigate these issues, including techniques like gradient penalty, label smoothing, and two-time scale updates. In this project, we will explore these techniques and implement them to stabilize GAN training.

We will demonstrate some stabilization techniques using a simple GAN model.

About:
✅ What It Does:
Gradient penalty is added to the discriminator's loss to stabilize the GAN training by penalizing the gradients that deviate from 1.

The model is trained using a min-max loss function for both the discriminator and generator.

Generates images and displays them periodically to visualize the progress of the GAN training.

Key features:
Gradient penalty helps to stabilize the training of GANs by ensuring smooth gradients for the discriminator.

Adversarial loss is combined with gradient penalty to balance the training between the generator and discriminator.

The model can generate fashion-like images after training, using the Fashion-MNIST dataset.
"""

# pip install torch torchvision matplotlib
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
 
# 1. Define the Generator and Discriminator for GAN
class Generator(nn.Module):
    def __init__(self, z_dim=100):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(z_dim, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, 1024)
        self.fc4 = nn.Linear(1024, 28 * 28)  # Output 28x28 image
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
 
    def forward(self, z):
        x = self.relu(self.fc1(z))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return self.tanh(x).view(-1, 1, 28, 28)  # Reshape to 28x28 image
 
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
 
# 2. Gradient Penalty for stabilizing GAN training
def compute_gradient_penalty(discriminator, real_samples, fake_samples, device):
    # Random weight term for interpolation between real and fake samples
    alpha = torch.rand(real_samples.size(0), 1, 1, 1).to(device)
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    
    # Discriminator output for interpolated images
    d_interpolates = discriminator(interpolates)
    
    # Compute the gradients of the output with respect to the interpolated images
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones(d_interpolates.size()).to(device),
        create_graph=True,
        retain_graph=True,
    )[0]
    
    # Compute the gradient penalty
    gradient_penalty = ((gradients.view(gradients.size(0), -1).norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty
 
# 3. Training the GAN model with gradient penalty
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
generator = Generator(z_dim=100).to(device)
discriminator = Discriminator().to(device)
 
# 4. Loss function and optimizer
adversarial_loss = nn.BCELoss()
optimizer_g = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_d = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
 
# 5. Load MNIST dataset for training
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
 
# 6. Training Loop
num_epochs = 50
for epoch in range(num_epochs):
    for i, (real_images, _) in enumerate(train_loader):
        real_images = real_images.to(device)
        
        # Create labels for real and fake data
        real_labels = torch.ones(real_images.size(0), 1).to(device)
        fake_labels = torch.zeros(real_images.size(0), 1).to(device)
 
        # Train the Discriminator
        optimizer_d.zero_grad()
 
        real_outputs = discriminator(real_images)
        d_loss_real = adversarial_loss(real_outputs, real_labels)
 
        z = torch.randn(real_images.size(0), 100).to(device)  # Random noise
        fake_images = generator(z)
        fake_outputs = discriminator(fake_images.detach())
        d_loss_fake = adversarial_loss(fake_outputs, fake_labels)
 
        # Compute the gradient penalty
        gradient_penalty = compute_gradient_penalty(discriminator, real_images, fake_images, device)
 
        # Total Discriminator loss
        d_loss = d_loss_real + d_loss_fake + 10 * gradient_penalty
        d_loss.backward()
        optimizer_d.step()
 
        # Train the Generator
        optimizer_g.zero_grad()
 
        fake_outputs = discriminator(fake_images)
        g_loss = adversarial_loss(fake_outputs, real_labels)
        g_loss.backward()
        optimizer_g.step()
 
    # Print loss every few epochs
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}')
 
    # Generate and display images every few epochs
    if (epoch + 1) % 10 == 0:
        with torch.no_grad():
            z = torch.randn(64, 100).to(device)
            fake_images = generator(z).cpu()
            grid_img = torchvision.utils.make_grid(fake_images, nrow=8, normalize=True)
            plt.imshow(grid_img.permute(1, 2, 0))
            plt.title(f"Generated Images at Epoch {epoch + 1}")
            plt.show()