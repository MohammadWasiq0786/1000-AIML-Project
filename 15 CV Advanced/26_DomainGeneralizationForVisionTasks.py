"""
Project 586: Domain Generalization for Vision Tasks
Description:
Domain generalization aims to improve the model's performance across multiple domains by learning domain-invariant features. This approach is useful when the model needs to generalize well to unseen domains, such as recognizing objects across different environments or lighting conditions. In this project, we will explore domain generalization techniques to help the model perform well on a variety of visual tasks across domains.
"""

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
 
# 1. Define a simple Domain-Adversarial Neural Network (DANN) for domain generalization
class DANN(nn.Module):
    def __init__(self):
        super(DANN, self).__init__()
        self.feature_extractor = torchvision.models.resnet18(pretrained=True)
        self.feature_extractor.fc = nn.Identity()  # Remove the final layer for feature extraction
        self.classifier = nn.Linear(512, 10)  # CIFAR-10 has 10 classes
        self.domain_classifier = nn.Linear(512, 2)  # Binary classification (source vs target domain)
 
    def forward(self, x):
        features = self.feature_extractor(x)
        class_output = self.classifier(features)
        domain_output = self.domain_classifier(features)
        return class_output, domain_output
 
# 2. Load source and target domain datasets (e.g., CIFAR-10 and SVHN for domain generalization)
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
 
source_data = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
target_data = torchvision.datasets.SVHN(root='./data', split='train', download=True, transform=transform)
 
source_loader = DataLoader(source_data, batch_size=32, shuffle=True)
target_loader = DataLoader(target_data, batch_size=32, shuffle=True)
 
# 3. Initialize DANN model, loss function, and optimizer
model = DANN()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0002)
 
# 4. Define a function to train the model with domain adversarial training
def train(model, source_loader, target_loader, criterion, optimizer, num_epochs=5):
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
 
        for (source_images, source_labels), (target_images, _) in zip(source_loader, target_loader):
            optimizer.zero_grad()
 
            # Forward pass on source and target domains
            class_output, domain_output = model(source_images)
            source_class_loss = criterion(class_output, source_labels)
            source_domain_loss = criterion(domain_output, torch.zeros(source_images.size(0)))  # Source domain label: 0
 
            # Forward pass on target domain (with pseudo-labels or domain-adversarial loss)
            _, target_domain_output = model(target_images)
            target_domain_loss = criterion(target_domain_output, torch.ones(target_images.size(0)))  # Target domain label: 1
 
            # Combine the losses
            loss = source_class_loss + source_domain_loss + target_domain_loss
            loss.backward()
            optimizer.step()
 
            total_loss += loss.item()
 
        print(f"Epoch {epoch+1}, Loss: {total_loss / len(source_loader)}")
 
# 5. Train the model
train(model, source_loader, target_loader, criterion, optimizer)
 
# 6. Visualize some images from source and target domains
source_image = source_images[0].permute(1, 2, 0).detach().numpy()
target_image = target_images[0].permute(1, 2, 0).detach().numpy()
 
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(source_image)
plt.title("Source Domain Image")
 
plt.subplot(1, 2, 2)
plt.imshow(target_image)
plt.title("Target Domain Image")
 
plt.show()