"""
Project 584: Semi-supervised Image Classification
Description:
Semi-supervised image classification involves using a small amount of labeled data alongside a large amount of unlabeled data to improve the performance of a model. In this project, we will implement a semi-supervised learning approach using pseudo-labeling or self-training techniques. The goal is to use the model's predictions on unlabeled data to enhance its training.
"""

import torch
import torchvision.transforms as transforms
from torchvision import models, datasets
from torch.utils.data import DataLoader, Subset
import torch.optim as optim
import numpy as np
from sklearn.metrics import accuracy_score
 
# 1. Load a small labeled dataset and a large unlabeled dataset
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
 
labeled_data = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
unlabeled_data = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
 
# 2. Subset labeled data for training (using a small portion of the dataset)
labeled_subset = Subset(labeled_data, np.arange(1000))  # Using 1000 labeled samples
unlabeled_subset = Subset(unlabeled_data, np.arange(5000))  # Using 5000 unlabeled samples
 
labeled_loader = DataLoader(labeled_subset, batch_size=32, shuffle=True)
unlabeled_loader = DataLoader(unlabeled_subset, batch_size=32, shuffle=False)
 
# 3. Load a pre-trained model (ResNet50) for semi-supervised learning
model = models.resnet50(pretrained=True)
model.fc = torch.nn.Linear(model.fc.in_features, 10)  # Adjust for CIFAR-10 classes
model.train()
 
# 4. Define the loss function and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
 
# 5. Train the model with pseudo-labeling
def pseudo_labeling_step(model, labeled_loader, unlabeled_loader, criterion, optimizer):
    for epoch in range(5):  # Run for 5 epochs
        model.train()
        for batch_idx, (inputs, labels) in enumerate(labeled_loader):
            # Train the model on the labeled dataset
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
 
        # Generate pseudo-labels for the unlabeled dataset
        model.eval()
        pseudo_labels = []
        with torch.no_grad():
            for inputs, _ in unlabeled_loader:
                outputs = model(inputs)
                pseudo_labels.append(torch.argmax(outputs, dim=1))
 
        pseudo_labels = torch.cat(pseudo_labels, dim=0)
 
        # Combine pseudo-labeled data with labeled data for retraining
        combined_data = torch.utils.data.ConcatDataset([labeled_subset, unlabeled_subset])  # Assuming the unlabeled data is now pseudo-labeled
 
    return model
 
# 6. Perform the pseudo-labeling step
model = pseudo_labeling_step(model, labeled_loader, unlabeled_loader, criterion, optimizer)
 
# 7. Evaluate the model on the labeled data
model.eval()
y_true = []
y_pred = []
with torch.no_grad():
    for inputs, labels in labeled_loader:
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        y_true.extend(labels.numpy())
        y_pred.extend(preds.numpy())
 
accuracy = accuracy_score(y_true, y_pred)
print(f"Accuracy on labeled data: {accuracy * 100:.2f}%")