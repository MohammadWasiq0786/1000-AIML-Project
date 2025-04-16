"""
Project 132. ResNet implementation
Description:
ResNet (Residual Network) revolutionized deep learning by introducing skip connections, allowing training of very deep networks without vanishing gradients. In this project, we implement a simplified version of ResNet (e.g., ResNet-18) using PyTorch, and test it on sample image data for classification tasks.

About:
🧠 What This Project Demonstrates:
Implements ResNet’s residual learning blocks

Builds a full ResNet-18 architecture from scratch

Outputs classification logits for input images
"""

# Install if not already: pip install torch torchvision matplotlib
 
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
 
# Basic building block: Residual block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
 
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
 
        self.downsample = downsample
 
    def forward(self, x):
        identity = x
 
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
 
        if self.downsample is not None:
            identity = self.downsample(x)
 
        out += identity
        return self.relu(out)
 
# ResNet Model (simplified version of ResNet-18)
class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10):
        super(ResNet, self).__init__()
        self.in_channels = 64
 
        self.conv = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
 
        self.layer1 = self.make_layer(block, 64,  layers[0])
        self.layer2 = self.make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self.make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self.make_layer(block, 512, layers[3], stride=2)
 
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
 
    def make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
 
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
 
        layers = [block(self.in_channels, out_channels, stride, downsample)]
        self.in_channels = out_channels
 
        for _ in range(1, blocks):
            layers.append(block(out_channels, out_channels))
 
        return nn.Sequential(*layers)
 
    def forward(self, x):
        x = self.relu(self.bn(self.conv(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
 
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)
 
# Create ResNet-18 instance
model = ResNet(ResidualBlock, [2, 2, 2, 2])
 
# Test with dummy input
dummy_input = torch.randn(1, 3, 224, 224)
output = model(dummy_input)
print("✅ Output shape:", output.shape)  # Expected: (1, 10)