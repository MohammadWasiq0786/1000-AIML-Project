"""
Project 222. Action recognition in videos
Description:
Action recognition identifies what action a person is performing in a video, such as walking, jumping, waving, or sitting. It combines spatial and temporal analysis of video frames. In this project, we'll use a pre-trained 3D CNN model from torchvision, such as ResNet3D (r3d_18), trained on the Kinetics-400 dataset, which can recognize 400+ human actions from short video clips.

About:
What It Does:
This project identifies human actions in short video clips. It’s essential for applications like smart surveillance, sports analytics, content tagging, gesture recognition, and video indexing. It can be extended for real-time video stream classification or multi-person action detection.
"""
# Install required packages:
# pip install torch torchvision opencv-python
 
import torch
import torchvision
import torchvision.transforms as transforms
import cv2
import numpy as np
from torchvision.io import read_video
 
# Load pre-trained 3D ResNet-18 model for action recognition
model = torchvision.models.video.r3d_18(pretrained=True)
model.eval()
 
# Load label map for Kinetics-400 dataset (simplified)
from urllib.request import urlopen
import json
 
# Load label names for Kinetics-400 actions
with urlopen("https://raw.githubusercontent.com/deepmind/kinetics-i3d/master/data/label_map.txt") as f:
    classes = [line.strip() for line in f.readlines()]
 
# Define transform for video frames
transform = transforms.Compose([
    transforms.Resize((112, 112)),         # Resize to model input
    transforms.Normalize([0.43216, 0.394666, 0.37645], 
                         [0.22803, 0.22145, 0.216989])  # Kinetics normalization
])
 
# Read a short video clip (3 seconds recommended, 30 frames)
video_path = 'action_sample.mp4'  # Replace with your video path
video, _, _ = read_video(video_path, pts_unit='sec')
video = video[:32]  # Take first 32 frames (3D CNN expects ~16–32 frames)
 
# Resize and normalize video frames
video = video.permute(0, 3, 1, 2) / 255.0  # (T, C, H, W)
video = transform(video)  # Apply normalization
video = video.unsqueeze(0)  # Add batch dimension: (1, T, C, H, W)
video = video.permute(0, 2, 1, 3, 4)  # Rearrange to (B, C, T, H, W)
 
# Predict the action
with torch.no_grad():
    outputs = model(video)
    predicted_idx = outputs.argmax(1).item()
 
# Show the predicted action
print("\n🎬 Predicted Action:")
print(classes[predicted_idx])