"""
Project 134. SimCLR for contrastive learning
Description:
SimCLR (Simple Framework for Contrastive Learning of Visual Representations) is a self-supervised method where a model learns to maximize agreement between different augmented views of the same image in latent space. It does this using contrastive loss without labeled data. In this project, we implement a simplified SimCLR framework using PyTorch on a small image dataset like CIFAR-10.

About:
🧠 What This Project Demonstrates:
Implements SimCLR’s contrastive learning framework

Learns useful image representations without labels

Uses a ResNet encoder and projection head
"""

# Install if not already: pip install torch torchvision matplotlib
 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
 
# Basic data augmentation for SimCLR
class SimCLRTransform:
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(size=32),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor()
        ])
    def __call__(self, x):
        return self.transform(x), self.transform(x)
 
# Dataset & DataLoader
transform = SimCLRTransform()
dataset = torchvision.datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=True, num_workers=2)
 
# Simple encoder backbone (ResNet-18) and projection head
from torchvision.models import resnet18
 
class SimCLR(nn.Module):
    def __init__(self, projection_dim=128):
        super(SimCLR, self).__init__()
        self.encoder = resnet18(pretrained=False)
        self.encoder.fc = nn.Identity()  # Remove classification head
 
        # Projection MLP (2-layer)
        self.projector = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, projection_dim)
        )
 
    def forward(self, x):
        h = self.encoder(x)        # Representation
        z = self.projector(h)      # Projection
        return F.normalize(z, dim=1)
 
# NT-Xent contrastive loss
def nt_xent_loss(z1, z2, temperature=0.5):
    z = torch.cat([z1, z2], dim=0)  # (2N, D)
    sim_matrix = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2)  # (2N, 2N)
 
    # Create labels
    N = z1.size(0)
    labels = torch.arange(N, device=z.device)
    labels = torch.cat([labels, labels], dim=0)
 
    # Mask self-similarity
    mask = torch.eye(2 * N, device=z.device).bool()
    sim_matrix = sim_matrix.masked_fill(mask, -9e15)
 
    # Scale by temperature and apply softmax
    sim_matrix /= temperature
    loss = F.cross_entropy(sim_matrix, labels)
    return loss
 
# Training SimCLR (1 epoch demo)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimCLR().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
 
model.train()
for batch in dataloader:
    (x1, x2), _ = batch
    x1, x2 = x1.to(device), x2.to(device)
 
    z1 = model(x1)
    z2 = model(x2)
 
    loss = nt_xent_loss(z1, z2)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(f"SimCLR Training Loss: {loss.item():.4f}")
    break  # Remove this to train full epoch