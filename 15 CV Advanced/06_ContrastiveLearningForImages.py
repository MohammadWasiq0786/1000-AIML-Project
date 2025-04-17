"""
Project 566: Contrastive Learning for Images
Description:
Contrastive learning is a self-supervised learning approach where the model learns by comparing similar and dissimilar pairs of images. The goal is to bring similar images (or views of the same object) closer together in the feature space and push dissimilar images further apart. In this project, we will implement contrastive learning using techniques like SimCLR or MoCo.
"""

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import datasets
import matplotlib.pyplot as plt
 
# 1. Define a simple SimCLR model (similar to the one used in self-supervised learning)
class SimCLR(nn.Module):
    def __init__(self):
        super(SimCLR, self).__init__()
        self.resnet = models.resnet18(pretrained=True)
        self.resnet.fc = nn.Identity()  # Remove the fully connected layer
        self.fc = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )
 
    def forward(self, x):
        x = self.resnet(x)
        x = self.fc(x)
        return x
 
# 2. Set up dataset and data loaders with augmentation for contrastive learning
transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
 
# Use CIFAR-10 for simplicity (replace with a more complex dataset if needed)
dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
 
# 3. Initialize the SimCLR model
model = SimCLR()
 
# 4. Define contrastive loss (NT-Xent loss)
def contrastive_loss(x1, x2, temperature=0.07):
    cosine_similarity = nn.functional.cosine_similarity(x1.unsqueeze(1), x2.unsqueeze(0), dim=-1)
    labels = torch.arange(x1.size(0)).long().to(x1.device)
    logits = cosine_similarity / temperature
    loss = nn.CrossEntropyLoss()(logits, labels)
    return loss
 
# 5. Set up optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
 
# 6. Training loop (simplified for illustration)
for epoch in range(5):  # 5 epochs for illustration
    model.train()
    total_loss = 0
    for images, _ in dataloader:
        optimizer.zero_grad()
 
        # Simulate augmentations: Here we use the same batch, but typically we'd augment each image
        x1, x2 = images, images  # Replace with different augmentations for each image pair
 
        # Forward pass
        z1, z2 = model(x1), model(x2)
 
        # Compute the contrastive loss
        loss = contrastive_loss(z1, z2)
        loss.backward()
        optimizer.step()
 
        total_loss += loss.item()
 
    print(f"Epoch {epoch+1}, Loss: {total_loss / len(dataloader)}")
 
# 7. Visualizing a sample image from the dataset (optional)
plt.imshow(images[0].numpy().transpose(1, 2, 0))
plt.title("Sample Image from Dataset")
plt.show()