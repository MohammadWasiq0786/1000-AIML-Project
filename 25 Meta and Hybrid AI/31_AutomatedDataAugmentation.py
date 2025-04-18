"""
Project 991: Automated Data Augmentation
Description
Data augmentation is a technique used to artificially expand the size of a dataset by generating modified versions of data points through transformations (e.g., rotations, translations, noise). Automated data augmentation techniques can help improve model generalization by increasing the diversity of training examples. In this project, we will implement automated data augmentation using deep learning techniques, where we automatically generate augmented data for a given task.

Key Concepts Covered:
Data Augmentation: Generating new data points by applying various transformations (e.g., rotations, flips, brightness adjustments) to the original dataset.

Albumentations: A fast and flexible library for image augmentation in machine learning tasks.

Automated Data Augmentation: Automatically applying a predefined set of augmentation techniques to generate new training data, helping to improve the model's robustness and generalization.
"""

import numpy as np
import pandas as pd
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
import albumentations as A
from albumentations.pytorch import ToTensorV2
 
# Define an example image dataset
class CustomDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform
 
    def __len__(self):
        return len(self.images)
 
    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
 
        # Apply the transformations if any
        if self.transform:
            image = self.transform(image=image)["image"]
        
        return image, label
 
# Define the automated data augmentation pipeline using albumentations
augmentation = A.Compose([
    A.RandomRotate90(p=0.5),
    A.Flip(p=0.5),
    A.Transpose(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.GaussianBlur(blur_limit=3, p=0.2),
    A.HueSaturationValue(p=0.3),
    A.CLAHE(clip_limit=2.0, p=0.3),
    A.Resize(256, 256),
    ToTensorV2(),  # Convert the image to tensor format suitable for PyTorch
])
 
# Load a sample image dataset (e.g., CIFAR-10)
from torchvision import datasets
 
# Download the CIFAR-10 dataset (or use your own images)
train_dataset = datasets.CIFAR10(root='./data', train=True, download=True)
 
# Convert images to numpy arrays for compatibility with albumentations
train_images = np.array([np.array(image) for image in train_dataset.data])
train_labels = train_dataset.targets
 
# Split into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(train_images, train_labels, test_size=0.2, random_state=42)
 
# Create custom dataset and DataLoader
train_data = CustomDataset(X_train, y_train, transform=augmentation)
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
 
# Visualize a few augmented images from the dataset
import matplotlib.pyplot as plt
 
def show_images(loader):
    data_iter = iter(loader)
    images, labels = next(data_iter)
    grid = torchvision.utils.make_grid(images[:8], nrow=4)
    plt.figure(figsize=(12, 12))
    plt.imshow(grid.permute(1, 2, 0))
    plt.show()
 
# Show some augmented images
show_images(train_loader)