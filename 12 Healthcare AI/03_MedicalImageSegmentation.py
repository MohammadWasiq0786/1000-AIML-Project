"""
Project 443. Medical image segmentation
Description:
Medical image segmentation involves identifying and outlining anatomical structures (e.g., tumors, lungs, organs) in scans like MRI, CT, or X-ray. It’s a crucial step in diagnostics, surgical planning, and radiotherapy. In this project, we’ll implement a U-Net architecture — the gold standard in biomedical image segmentation.

About:
✅ What It Does:
Implements a U-Net to segment circular "tumors" in fake medical scans.

Trains using binary cross-entropy loss.

Visualizes the segmented output.

Ready to be applied to real-world datasets with CT/MRI/X-ray masks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
 
# 1. Define U-Net architecture
class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        def CBR(in_ch, out_ch):  # Conv-BN-ReLU block
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            )
        self.enc1 = nn.Sequential(CBR(1, 64), CBR(64, 64))
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = nn.Sequential(CBR(64, 128), CBR(128, 128))
        self.pool2 = nn.MaxPool2d(2)
 
        self.bottleneck = nn.Sequential(CBR(128, 256), CBR(256, 256))
 
        self.up1 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec1 = nn.Sequential(CBR(256, 128), CBR(128, 128))
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = nn.Sequential(CBR(128, 64), CBR(64, 64))
 
        self.out = nn.Conv2d(64, 1, 1)  # 1 output channel for binary mask
 
    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        b = self.bottleneck(self.pool2(e2))
        d1 = self.dec1(torch.cat([self.up1(b), e2], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d1), e1], dim=1))
        return torch.sigmoid(self.out(d2))
 
# 2. Simulated dataset (binary masks on synthetic blobs)
class SyntheticMedicalDataset(Dataset):
    def __init__(self, size=64, count=100):
        self.size = size
        self.count = count
 
    def __len__(self):
        return self.count
 
    def __getitem__(self, idx):
        image = np.zeros((self.size, self.size), dtype=np.float32)
        mask = np.zeros_like(image)
 
        # Random circle blob
        x, y = np.random.randint(10, self.size - 10, size=2)
        r = np.random.randint(5, 10)
        for i in range(self.size):
            for j in range(self.size):
                if (i - x)**2 + (j - y)**2 <= r**2:
                    image[i, j] = 1.0
                    mask[i, j] = 1.0
 
        image = torch.tensor(image).unsqueeze(0)
        mask = torch.tensor(mask).unsqueeze(0)
        return image, mask
 
# 3. Setup
dataset = SyntheticMedicalDataset()
loader = DataLoader(dataset, batch_size=8, shuffle=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.BCELoss()
 
# 4. Train loop
for epoch in range(1, 6):
    model.train()
    for images, masks in loader:
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, masks)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
 
# 5. Visualize example
model.eval()
img, mask = dataset[0]
pred = model(img.unsqueeze(0).to(device)).detach().cpu().squeeze().numpy()
 
plt.subplot(1, 3, 1)
plt.title("Image")
plt.imshow(img.squeeze(), cmap='gray')
plt.subplot(1, 3, 2)
plt.title("Ground Truth")
plt.imshow(mask.squeeze(), cmap='gray')
plt.subplot(1, 3, 3)
plt.title("Predicted Mask")
plt.imshow(pred > 0.5, cmap='gray')
plt.show()