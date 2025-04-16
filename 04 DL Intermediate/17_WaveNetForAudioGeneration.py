"""
Project 137. WaveNet for audio generation
Description:
WaveNet is a deep generative model developed by DeepMind that produces raw audio waveforms using causal and dilated convolutions. It generates realistic speech and audio one sample at a time by learning the probability distribution of audio samples. In this project, we implement a simplified version of WaveNet using PyTorch to generate synthetic audio.

About:
🧠 What This Project Demonstrates:
Implements a dilated causal convolutional WaveNet-like model

Predicts the next audio sample using past context

Trains on a sine wave, but works with any raw waveform data
"""

# Install if not already: pip install torch matplotlib numpy
 
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
 
# Hyperparameters
sample_rate = 8000      # 8kHz for simplicity
duration = 1            # seconds
samples = sample_rate * duration
channels = 1
quantization_channels = 256
 
# One-hot encoding for quantized audio
def one_hot_encode(x, num_classes):
    x = x.long()
    return F.one_hot(x, num_classes=num_classes).float()
 
# Dummy training waveform (sine wave)
t = np.linspace(0, duration, samples, endpoint=False)
wave = 0.5 * np.sin(2 * np.pi * 440 * t)  # 440Hz A-note
wave = ((wave + 1) / 2 * 255).astype(np.int64)  # Quantize to 0-255
 
# WaveNet model definition
class WaveNet(nn.Module):
    def __init__(self, residual_channels=32, dilation_depth=4, kernel_size=2):
        super(WaveNet, self).__init__()
        self.dilated_convs = nn.ModuleList()
        self.residuals = nn.ModuleList()
 
        for d in range(dilation_depth):
            dilation = 2 ** d
            self.dilated_convs.append(nn.Conv1d(residual_channels, 2 * residual_channels, kernel_size, dilation=dilation, padding=dilation))
            self.residuals.append(nn.Conv1d(residual_channels, residual_channels, 1))
 
        self.input_conv = nn.Conv1d(quantization_channels, residual_channels, 1)
        self.output = nn.Sequential(
            nn.ReLU(),
            nn.Conv1d(residual_channels, quantization_channels, 1)
        )
 
    def forward(self, x):
        x = self.input_conv(x)
        for conv, res in zip(self.dilated_convs, self.residuals):
            out = conv(x)
            tanh_out, sigm_out = torch.chunk(out, 2, dim=1)
            z = torch.tanh(tanh_out) * torch.sigmoid(sigm_out)
            x = res(z) + x  # residual connection
        return self.output(x)
 
# Training (for demo only – real training requires much more data/time)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = WaveNet().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()
 
# Prepare input and target
wave_tensor = torch.tensor(wave, dtype=torch.long).unsqueeze(0).to(device)  # (1, T)
x_input = one_hot_encode(wave_tensor[:, :-1], quantization_channels).permute(0, 2, 1)  # (1, C, T-1)
y_target = wave_tensor[:, 1:]  # Predict next sample
 
# Forward and train
model.train()
output = model(x_input)
loss = criterion(output.squeeze(0).permute(1, 0), y_target.squeeze(0))
optimizer.zero_grad()
loss.backward()
optimizer.step()
print(f"WaveNet Training Loss (1 step): {loss.item():.4f}")