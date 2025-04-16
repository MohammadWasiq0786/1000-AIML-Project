"""
Project 146. Action recognition in videos
Description:
Action recognition refers to identifying and classifying actions (e.g., jumping, waving, running) from video sequences. This project uses a pretrained deep learning model for action recognition, leveraging spatiotemporal patterns in frames. We'll use TorchVision's pre-trained models (e.g., r3d_18) trained on the Kinetics-400 dataset for real-world video action classification.

About:
🧠 What This Project Demonstrates:
Leverages a pretrained action recognition model (R(2+1)D-18)

Processes videos using temporal frame sampling

Classifies actions with output probabilities over 400 real-world activities
"""

# Install dependencies:
# pip install torch torchvision opencv-python
 
import torch
import torchvision
import torchvision.transforms as T
import cv2
import numpy as np
import os
 
# Load pretrained action recognition model
model = torchvision.models.video.r3d_18(pretrained=True)
model.eval()
 
# Class labels for Kinetics-400 (partial demo, real list has 400 classes)
kinetics_classes = ["abseiling", "air drumming", "answering questions", "applauding", "applying cream",
                    "archery", "arm wrestling", "arranging flowers", "assembling computer", "auctioning"]
 
# Video loader and preprocessor
def load_video_frames(path, num_frames=16, size=(112, 112)):
    cap = cv2.VideoCapture(path)
    frames = []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(total // num_frames, 1)
 
    for i in range(num_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i * step)
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, size)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()
 
    # Transform and normalize
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.43216, 0.394666, 0.37645],
                    std=[0.22803, 0.22145, 0.216989])
    ])
    frames = [transform(frame) for frame in frames]
    video = torch.stack(frames).permute(1, 0, 2, 3)  # [C, T, H, W]
    return video.unsqueeze(0)  # [1, C, T, H, W]
 
# Path to your video file
video_path = "sample_action.mp4"  # Replace with real video path
video_tensor = load_video_frames(video_path)
 
# Predict action
with torch.no_grad():
    outputs = model(video_tensor)
    probs = torch.nn.functional.softmax(outputs[0], dim=0)
    top5 = torch.topk(probs, k=5)
 
# Print top predictions
print("🎬 Top 5 Predicted Actions:")
for idx in top5.indices:
    print(f"{kinetics_classes[idx]} ({probs[idx]*100:.2f}%)")