"""
Project 361. Variational autoencoder implementation
Description:
A Variational Autoencoder (VAE) is a generative model used to generate new data similar to the training dataset. It is an extension of the autoencoder that introduces variational inference to learn the distribution of the data. VAEs are commonly used in image generation, anomaly detection, and unsupervised learning.

In this project, we’ll implement a VAE that learns a latent variable representation of the data and generates new samples from this learned distribution.

About:
✅ What It Does:
Defines a Variational Autoencoder (VAE) architecture with an encoder, decoder, and reparameterization trick for latent variable sampling

Loss function combines reconstruction loss (BCE) and KL divergence for regularization, ensuring the learned distribution approximates a Gaussian distribution

Trains on the MNIST dataset, learns a latent representation of the digits, and can generate new samples from the learned distribution
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt
 
# 1. Define the VAE architecture
class VAE(nn.Module):
    def __init__(self, z_dim=20):
        super(VAE, self).__init__()
        self.z_dim = z_dim
        
        # Encoder
        self.fc1 = nn.Linear(784, 400)
        self.fc21 = nn.Linear(400, z_dim)  # Mean of z
        self.fc22 = nn.Linear(400, z_dim)  # Log-variance of z
        
        # Decoder
        self.fc3 = nn.Linear(z_dim, 400)
        self.fc4 = nn.Linear(400, 784)
        
    def encode(self, x):
        h1 = torch.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std
    
    def decode(self, z):
        h3 = torch.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))
    
    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
 
# 2. Loss function for VAE
def loss_function(recon_x, x, mu, logvar):
    BCE = nn.BCELoss(reduction='sum')(recon_x, x.view(-1, 784))
    # KL divergence between the learned distribution and the prior
    # Regularizing the latent space
    # D_KL = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    # where mu, sigma are the learned mean and standard deviation of the latent variable
    # This encourages the learned distribution to be close to the standard normal distribution
    # for smooth and consistent sampling.
    # In simple terms, it penalizes the encoder from outputting too large/small of variance
    # or high deviations in latent space.
    # It ensures the latent space stays close to Gaussian distribution.
    # More about variational inference here: https://arxiv.org/abs/1312.6114
    # https://dl.acm.org/doi/10.1145/3065386
    # https://www.jmlr.org/papers/volume15/kingma14a/kingma14a.pdf
    # reference: https://pytorch.org/tutorials/beginner/vae.html
    # We are calculating the Kullback-Leibler divergence between the learned distribution
    # (q(z|x)) and a prior distribution (usually chosen as N(0,I))
    # Note that the prior is not trainable.
    # The learned distribution q(z|x) comes from the encoder, so its parameters (mean, variance)
    # depend on the input data.
    # The prior on the other hand, is fixed and is independent of x.
    # The code implements this via logvar (log variance).
    # z ~ N(0, I)
    # Both mu (mean) and logvar are the outputs of the encoder network. We can think of this as variational inference.
    # In practice, we use the reparameterization trick (sampling z from N(0, I)).
    # D_KL divergence encourages the encoder to output a distribution close to N(0, I).
    # This is added to the reconstruction error (BCE loss) during optimization.
    # Overall loss = Reconstruction loss + KL Divergence loss (regularization term).
    # Note that the BCE term tries to match the distribution to the data and the KL term tries to keep it close to N(0, I).
    # This means that the model will learn how to generate new, similar data while keeping the latent space organized.
 
    # KL divergence term
    # Equation for KL divergence
    # eq. (9) in Kingma et al.
    # D_KL = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    # print(f"BCE: {BCE:.4f}, KL: {KL:.4f}")
    # KL divergence loss
    # sum(log(sigma^2) - mu^2 - sigma^2 + 1)
    # mu and logvar are outputs of encoder
    # note, using the Gaussian prior
    # q(z|x) is the encoder, p(z) is the prior, and D_KL is the divergence between the two
    # Tying the encoder's output to the prior by minimizing the KL divergence term
    # that ensures the learned distribution stays close to the prior
    # we want the decoder to be able to generate from the learned latent distribution
 
    # sum(1 + log(sigma^2) - mu^2 - sigma^2)
    # Returning loss that combines reconstruction and regularization
    # KLD term (KL divergence term is penalized)
    # full loss:
    # L_total = L_reconstruction + L_kl
    # See also Eq. (12) in Kingma et al.
    # ref: https://pytorch.org/tutorials/beginner/vae.html
    # more on VAE from https://arxiv.org/abs/1312.6114
    # see equation (2) of Kingma et al. for reconstruction and regularization terms
    # more about VAEs: https://arxiv.org/abs/1907.09711
    # finally: loss function = BCE + KL divergence
    # This produces the final sum
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD
 
# 3. Set up training loop
def train_vae():
    # Load MNIST dataset
    transform = transforms.ToTensor()
    train_dataset = MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    
    # Initialize model, optimizer
    model = VAE()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    # Training loop
    epochs = 10
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch_idx, (data, _) in enumerate(train_loader):
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(data)
            loss = loss_function(recon_batch, data, mu, logvar)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {train_loss / len(train_loader.dataset):.4f}")
 
# 4. Train the VAE
train_vae()