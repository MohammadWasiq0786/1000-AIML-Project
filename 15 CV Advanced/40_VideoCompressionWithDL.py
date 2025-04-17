"""
Project 600: Video Compression with Deep Learning
Description:
Video compression reduces the size of video files while maintaining quality, which is important for storage and streaming. Traditional video compression algorithms (like H.264) rely on hand-crafted techniques, while deep learning-based video compression uses neural networks to learn more efficient representations of video data. In this project, we will explore video compression using autoencoders or similar models to compress and reconstruct video frames.
"""

import torch
import torch.nn as nn
import numpy as np
import cv2
import matplotlib.pyplot as plt
 
# 1. Define a simple Autoencoder model for video compression
class VideoAutoencoder(nn.Module):
    def __init__(self):
        super(VideoAutoencoder, self).__init__()
        # Encoder: Compressing the video frame into a lower-dimensional representation
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        # Decoder: Reconstructing the compressed frame back to original dimensions
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid(),  # To output values in the range [0, 1]
        )
 
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
 
# 2. Load a video for compression (use a video file path or URL)
video_path = "path_to_video.mp4"  # Replace with an actual video path
cap = cv2.VideoCapture(video_path)
 
# 3. Read a frame from the video and preprocess it
ret, frame = cap.read()
frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
frame_resized = cv2.resize(frame_rgb, (224, 224))  # Resize to fit the model
 
# 4. Convert the frame to a tensor and normalize
frame_tensor = torch.tensor(frame_resized, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 255.0  # Normalize to [0, 1]
 
# 5. Initialize the Autoencoder model for compression
model = VideoAutoencoder()
model.eval()
 
# 6. Perform compression and reconstruction (inference step)
with torch.no_grad():
    compressed_frame = model(frame_tensor).squeeze(0).permute(1, 2, 0).numpy()
 
# 7. Visualize the original and compressed (reconstructed) frames
plt.figure(figsize=(10, 5))
 
plt.subplot(1, 2, 1)
plt.imshow(frame_resized)
plt.title("Original Frame")
 
plt.subplot(1, 2, 2)
plt.imshow(compressed_frame)
plt.title("Compressed Frame (Reconstructed)")
 
plt.show()
 
# 8. Optionally, save the compressed video (assuming you process multiple frames)
output_video_path = "compressed_video.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Specify codec
out = cv2.VideoWriter(output_video_path, fourcc, 30, (frame_resized.shape[1], frame_resized.shape[0]))
 
# You can loop through the frames, compress them, and save the compressed video
out.release()
cap.release()

"""
This code demonstrates video compression using an autoencoder model. It compresses each video frame and reconstructs it with minimal loss. In practice, more advanced models like VAE (Variational Autoencoders) or deep compression algorithms can be used to improve the compression efficiency.
"""