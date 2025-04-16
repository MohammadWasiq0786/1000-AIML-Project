"""
Project 231. Image denoising
Description:
Image denoising involves removing unwanted noise or grain from images while preserving essential details. It’s crucial for medical imaging, photography, security cameras, and low-light enhancement. In this project, we'll use a pre-trained DnCNN (Denoising CNN) model to clean up noisy images using deep learning.

About:
What It Does:
This project effectively removes random noise from corrupted images, producing a clean version while preserving fine details. It’s widely used in X-ray enhancement, astrophotography, scanned document cleanup, and night photography tools. For real-world use, try pretrained models like FFDNet, DRUNet, or BM3D for color denoising and batch processing.
"""
# Install required packages:
# pip install torch torchvision matplotlib pillow
 
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
 
# Define a simplified DnCNN architecture
class DnCNN(nn.Module):
    def __init__(self, channels=1, num_of_layers=17):
        super(DnCNN, self).__init__()
        layers = [nn.Conv2d(channels, 64, kernel_size=3, padding=1), nn.ReLU(inplace=True)]
        for _ in range(num_of_layers - 2):
            layers.append(nn.Conv2d(64, 64, kernel_size=3, padding=1))
            layers.append(nn.BatchNorm2d(64))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(64, channels, kernel_size=3, padding=1))
        self.dncnn = nn.Sequential(*layers)
 
    def forward(self, x):
        out = self.dncnn(x)
        return x - out  # Residual learning: input - noise
 
# Load the pre-trained model (you can download trained weights or train on BSD400)
model = DnCNN()
model.load_state_dict(torch.load("dncnn_gray.pth", map_location="cpu"))  # Replace with your .pth file
model.eval()
 
# Load a noisy grayscale image
image_path = "noisy_image.jpg"  # Replace with your noisy image
img = Image.open(image_path).convert("L")  # Convert to grayscale
 
# Transform image to tensor
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize to [-1, 1]
])
input_tensor = transform(img).unsqueeze(0)
 
# Denoise the image
with torch.no_grad():
    output_tensor = model(input_tensor)
 
# Convert output to displayable format
output_tensor = output_tensor.squeeze().clamp(-1, 1) * 0.5 + 0.5  # Reverse normalization
output_img = transforms.ToPILImage()(output_tensor)
 
# Display original and denoised image
plt.figure(figsize=(10, 5))
 
plt.subplot(1, 2, 1)
plt.imshow(img, cmap='gray')
plt.title("Noisy Image")
plt.axis('off')
 
plt.subplot(1, 2, 2)
plt.imshow(output_img, cmap='gray')
plt.title("Denoised Output")
plt.axis('off')
 
plt.tight_layout()
plt.show()