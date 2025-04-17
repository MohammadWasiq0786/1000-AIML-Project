"""
Project 598: Video Denoising
Description:
Video denoising is the process of removing noise from video frames, improving their quality and clarity. This is important in low-light video conditions or when dealing with compressed videos. In this project, we will implement a video denoising model using deep learning-based approaches such as autoencoders or convolutional neural networks (CNNs).
"""

import torch
import torch.nn as nn
import cv2
import numpy as np
import matplotlib.pyplot as plt
 
# 1. Define a simple autoencoder model for video denoising
class DenoisingAutoencoder(nn.Module):
    def __init__(self):
        super(DenoisingAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()  # To ensure output is in the range [0, 1]
        )
 
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
 
# 2. Load a pre-trained model (for the sake of simplicity, we use a randomly initialized model here)
model = DenoisingAutoencoder()
model.eval()
 
# 3. Load a video (use a video file path or URL)
video_path = "path_to_video.mp4"  # Replace with an actual video path
cap = cv2.VideoCapture(video_path)
 
# 4. Process each frame of the video and apply denoising
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
 
    # 5. Convert frame to tensor and normalize
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_tensor = torch.tensor(frame_rgb, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 255.0  # Normalize to [0, 1]
 
    # 6. Apply denoising using the model
    with torch.no_grad():
        denoised_frame = model(frame_tensor).squeeze(0).permute(1, 2, 0).numpy()
 
    # 7. Display the original and denoised frames
    plt.figure(figsize=(10, 5))
 
    plt.subplot(1, 2, 1)
    plt.imshow(frame_rgb)
    plt.title("Original Frame")
 
    plt.subplot(1, 2, 2)
    plt.imshow(denoised_frame)
    plt.title("Denoised Frame")
 
    plt.show()
 
# 8. Release the video capture object
cap.release()