"""
Project 585: Unsupervised Domain Adaptation for Vision
Description:
Unsupervised domain adaptation involves transferring knowledge learned from a source domain (with labeled data) to a target domain (with no labeled data). The goal is to make a model perform well on the target domain without requiring annotated data. In this project, we will use an unsupervised domain adaptation technique such as CycleGAN or DANN (Domain-Adversarial Neural Network) to adapt a model trained on one dataset (source domain) to work well on another dataset (target domain).
"""

import torch
from torch import nn
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision import models
import matplotlib.pyplot as plt
 
# 1. Load source and target domain datasets (e.g., different styles of images)
source_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
 
target_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
 
source_data = ImageFolder("path_to_source_domain_images", transform=source_transform)
target_data = ImageFolder("path_to_target_domain_images", transform=target_transform)
 
source_loader = DataLoader(source_data, batch_size=32, shuffle=True)
target_loader = DataLoader(target_data, batch_size=32, shuffle=True)
 
# 2. Define a simple CycleGAN-based domain adaptation model
class CycleGAN(nn.Module):
    def __init__(self):
        super(CycleGAN, self).__init__()
        # Define generator and discriminator models (simplified example)
        self.generator = models.resnet50(pretrained=True)  # Example model
        self.discriminator = models.resnet50(pretrained=True)
 
    def forward(self, x):
        generated = self.generator(x)  # Generate output based on input
        return generated
 
# 3. Initialize CycleGAN model, loss function, and optimizer
model = CycleGAN()
criterion = nn.MSELoss()  # Example loss function for adaptation
optimizer = torch.optim.Adam(model.parameters(), lr=0.0002, betas=(0.5, 0.999))
 
# 4. Train the model (simplified training loop for domain adaptation)
for epoch in range(5):  # For simplicity, train for 5 epochs
    model.train()
    for (source_images, _), (target_images, _) in zip(source_loader, target_loader):
        optimizer.zero_grad()
 
        # Perform a forward pass on the source and target domains
        source_output = model(source_images)
        target_output = model(target_images)
 
        # Calculate the loss and update the model
        loss = criterion(source_output, target_output)
        loss.backward()
        optimizer.step()
 
    print(f"Epoch {epoch+1}, Loss: {loss.item()}")
 
# 5. Visualize a few images from both domains (source and target)
source_image = source_images[0].permute(1, 2, 0).detach().numpy()
target_image = target_images[0].permute(1, 2, 0).detach().numpy()
 
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(source_image)
plt.title("Source Image")
 
plt.subplot(1, 2, 2)
plt.imshow(target_image)
plt.title("Target Image")
 
plt.show()