"""
Project 216. Signature verification system
Description:
Signature verification is the process of validating whether a given signature is genuine or forged. It’s commonly used in banking, legal documents, and biometric authentication. In this project, we’ll implement a basic offline signature verification system using a Siamese neural network architecture that compares two signature images and determines their similarity.

About:
What It Does:
This project compares two signatures and evaluates how visually similar they are using deep features. It can be expanded for real-world datasets, threshold tuning, and training a full Siamese network on a signature dataset like CEDAR or GPDS, making it suitable for secure and automated signature verification systems.
"""
# Install dependencies:
# pip install torch torchvision matplotlib opencv-python
 
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import datasets
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import numpy as np
 
# Define a simple CNN-based feature extractor for signatures
class SignatureNet(nn.Module):
    def __init__(self):
        super(SignatureNet, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 64 * 64, 256),  # Assume resized to (256, 256)
            nn.ReLU()
        )
        
    def forward(self, x):
        x = self.cnn(x)
        x = self.fc(x)
        return x
 
# Load and preprocess signature images (convert to grayscale and tensor)
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.Grayscale(),               # Convert to single channel
    transforms.ToTensor(),                # Convert to PyTorch tensor
    transforms.Normalize([0.5], [0.5])    # Normalize grayscale values
])
 
# Load two signature samples
img1 = Image.open("signature_real.png")    # Replace with real signature image
img2 = Image.open("signature_test.png")    # Replace with test signature image
 
# Apply transformations
img1_tensor = transform(img1).unsqueeze(0)
img2_tensor = transform(img2).unsqueeze(0)
 
# Load model (initialize and simulate pre-trained weights)
model = SignatureNet()
model.eval()
 
# Get feature embeddings
with torch.no_grad():
    emb1 = model(img1_tensor)
    emb2 = model(img2_tensor)
 
# Compute cosine similarity between embeddings
cos = nn.CosineSimilarity(dim=1, eps=1e-6)
similarity = cos(emb1, emb2).item()
 
# Display both signatures side by side
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.title("Signature 1 (Reference)")
plt.imshow(img1, cmap='gray')
plt.axis('off')
 
plt.subplot(1, 2, 2)
plt.title("Signature 2 (To Verify)")
plt.imshow(img2, cmap='gray')
plt.axis('off')
plt.suptitle(f"Similarity Score: {similarity:.4f}", fontsize=14)
plt.show()
 
# Print result
print("\n🔐 Verification Result:")
if similarity > 0.85:
    print("✅ Signatures Match (Genuine)")
else:
    print("❌ Signatures Do Not Match (Possible Forgery)")