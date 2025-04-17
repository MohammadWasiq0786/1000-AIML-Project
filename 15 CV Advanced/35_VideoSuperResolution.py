"""
Project 595: Video Super-Resolution
Description:
Video super-resolution involves enhancing the resolution of video frames to make them sharper and more detailed. This process can improve the quality of videos, especially when working with low-resolution content. In this project, we will apply super-resolution techniques such as deep learning-based models (e.g., SRCNN or VDSR) to upscale low-resolution video frames.
"""

import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import matplotlib.pyplot as plt
 
# 1. Define the SRCNN (Super-Resolution Convolutional Neural Network) model
class SRCNN(nn.Module):
    def __init__(self):
        super(SRCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=9, padding=4)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=1)
        self.conv3 = nn.Conv2d(32, 1, kernel_size=5, padding=2)
 
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.conv3(x)
        return x
 
# 2. Load a pre-trained SRCNN model (for demonstration purposes, you can use a custom-trained model)
model = SRCNN()
model.load_state_dict(torch.load('srcnn_model.pth'))  # Assuming model is pre-trained
model.eval()
 
# 3. Load the video (use a video file path or URL)
video_path = "path_to_video.mp4"  # Replace with an actual video path
cap = cv2.VideoCapture(video_path)
 
# 4. Process each frame of the video and apply super-resolution
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # 5. Convert frame to grayscale and normalize
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_tensor = torch.tensor(gray_frame, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
    frame_tensor = frame_tensor / 255.0  # Normalize to [0, 1]
 
    # 6. Apply SRCNN to upscale the image
    with torch.no_grad():
        upscaled_frame = model(frame_tensor).squeeze(0).squeeze(0).numpy()
 
    # 7. Convert the upscaled frame back to the original frame size
    upscaled_frame = (upscaled_frame * 255).astype(np.uint8)
    upscaled_frame = cv2.resize(upscaled_frame, (frame.shape[1], frame.shape[0]))
 
    # 8. Visualize the original and upscaled frames
    plt.figure(figsize=(10, 5))
 
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    plt.title("Original Frame")
 
    plt.subplot(1, 2, 2)
    plt.imshow(upscaled_frame, cmap='gray')
    plt.title("Upscaled Frame")
 
    plt.show()
 
# 9. Release the video capture object
cap.release()