"""
Project 233. Depth estimation from monocular images
Description:
Depth estimation from a single (monocular) image involves predicting distance information for each pixel, creating a depth map that simulates 3D perception. This is essential in AR/VR, robotics, 3D reconstruction, and autonomous driving. In this project, we’ll use a pre-trained MiDaS model (by Intel) to generate depth maps from 2D images.

About:
What It Does:
This project estimates a realistic depth map from a single 2D image, enabling applications like 3D scene modeling, obstacle detection, bokeh effect simulation, and robot vision. MiDaS works surprisingly well on diverse scenes, and you can improve accuracy with MiDaS v3-large or fuse it with SLAM for real-time applications.
"""
# Install required packages:
# pip install torch torchvision opencv-python matplotlib
 
import torch
import cv2
import matplotlib.pyplot as plt
import numpy as np
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
 
# Load the pre-trained MiDaS model from PyTorch Hub
midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")  # Lightweight version
midas.eval()
 
# Move model to device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
midas.to(device)
 
# Define input transformation (resize, normalize, tensorize)
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = midas_transforms.small_transform
 
# Load and prepare the image
image_path = "scene.jpg"  # Replace with your image
img = cv2.imread(image_path)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
 
# Apply preprocessing
input_tensor = transform(img_rgb).to(device)
 
# Inference
with torch.no_grad():
    prediction = midas(input_tensor.unsqueeze(0))  # Add batch dim
    prediction = torch.nn.functional.interpolate(
        prediction.unsqueeze(1),
        size=img_rgb.shape[:2],
        mode="bicubic",
        align_corners=False
    ).squeeze()
 
# Normalize depth map to 0-1 range
depth_map = prediction.cpu().numpy()
depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
 
# Display original image and estimated depth map
plt.figure(figsize=(12, 5))
 
plt.subplot(1, 2, 1)
plt.imshow(img_rgb)
plt.title("Original Image")
plt.axis('off')
 
plt.subplot(1, 2, 2)
plt.imshow(depth_map, cmap='inferno')
plt.title("Estimated Depth Map")
plt.axis('off')
 
plt.tight_layout()
plt.show()