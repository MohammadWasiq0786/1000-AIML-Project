"""
Project 131. U-Net for image segmentation
Description:
U-Net is a powerful convolutional neural network architecture designed for semantic segmentation, especially in biomedical image analysis. It uses a contracting path to capture context and a symmetric expanding path for precise localization. In this project, we implement a U-Net using PyTorch to segment objects in images (e.g., cells, roads, buildings).

About:
🧠 What This Project Demonstrates:
Implements the U-Net architecture with skip connections

Produces pixel-wise segmentation maps from input images

Uses transpose convolutions to upsample during decoding
"""

# Install if not already: pip install torch torchvision matplotlib
 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.datasets import VOCSegmentation
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
 
# Define U-Net architecture
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        def conv_block(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, 3, padding=1),
                nn.ReLU(inplace=True)
            )
 
        self.enc1 = conv_block(3, 64)
        self.enc2 = conv_block(64, 128)
        self.enc3 = conv_block(128, 256)
 
        self.pool = nn.MaxPool2d(2)
 
        self.bottleneck = conv_block(256, 512)
 
        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = conv_block(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = conv_block(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = conv_block(128, 64)
 
        self.out_conv = nn.Conv2d(64, 1, 1)  # Binary segmentation
 
    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
 
        # Bottleneck
        b = self.bottleneck(self.pool(e3))
 
        # Decoder
        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
 
        return torch.sigmoid(self.out_conv(d1))
 
# Dummy input test
model = UNet()
dummy_input = torch.randn(1, 3, 128, 128)  # Simulated RGB image
output = model(dummy_input)
print("✅ Output shape:", output.shape)  # Should be (1, 1, 128, 128)
 
# Visualize input and output
plt.subplot(1, 2, 1)
plt.imshow(dummy_input[0].permute(1, 2, 0).detach().numpy())
plt.title("Input Image")
plt.axis("off")
 
plt.subplot(1, 2, 2)
plt.imshow(output[0][0].detach().numpy(), cmap='gray')
plt.title("Predicted Segmentation Mask")
plt.axis("off")
plt.tight_layout()
plt.show()