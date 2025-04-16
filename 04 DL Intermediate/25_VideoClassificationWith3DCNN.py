"""
Project 145. Video classification with 3D CNN
Description:
Video classification involves identifying actions, events, or objects in a video sequence. Unlike image classification, it requires understanding both spatial and temporal features. This project uses a 3D Convolutional Neural Network (3D CNN), which learns directly from the video clip by applying convolutions across both space and time dimensions.

About:
🧠 What This Project Demonstrates:
Implements a 3D convolutional network to extract spatio-temporal features

Handles video input as a 5D tensor: [Batch, Channels, Time, Height, Width]

Uses OpenCV for frame extraction and PyTorch for modeling
"""

# Install if not already: pip install torch torchvision opencv-python
 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import cv2
import numpy as np
import os
 
# Define a basic 3D CNN
class Simple3DCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(Simple3DCNN, self).__init__()
        self.conv1 = nn.Conv3d(3, 32, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool3d(2)
        self.conv2 = nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool3d(2)
        self.fc1 = nn.Linear(64 * 4 * 28 * 28, 256)
        self.fc2 = nn.Linear(256, num_classes)
 
    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))  # -> [B, 32, T/2, H/2, W/2]
        x = self.pool2(F.relu(self.conv2(x)))  # -> [B, 64, T/4, H/4, W/4]
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)
 
# Preprocessing function: extract frames & resize
def load_video_tensor(video_path, num_frames=16, resize=(112, 112)):
    cap = cv2.VideoCapture(video_path)
    frames = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(total_frames // num_frames, 1)
 
    for i in range(num_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i * step)
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, resize)
        frame = frame[:, :, ::-1]  # BGR to RGB
        frames.append(frame)
    cap.release()
 
    # Normalize and convert to tensor
    frames = np.stack(frames).astype(np.float32) / 255.0
    frames = np.transpose(frames, (3, 0, 1, 2))  # [C, T, H, W]
    return torch.tensor(frames).unsqueeze(0)  # [1, C, T, H, W]
 
# Example usage
video_path = 'sample_video.mp4'  # Replace with a short test video
video_tensor = load_video_tensor(video_path)  # [1, 3, 16, 112, 112]
 
# Initialize model and predict
model = Simple3DCNN(num_classes=5)  # Assume 5 actions
model.eval()
with torch.no_grad():
    outputs = model(video_tensor)
    prediction = torch.argmax(outputs, dim=1).item()
    print(f"🎥 Predicted Class ID: {prediction}")