"""
Project 300. Temporal convolutional networks
Description:
Temporal Convolutional Networks (TCNs) are a modern alternative to RNNs for modeling sequential data. They rely on:

1D causal convolutions (no future leakage)

Dilations to capture long-term dependencies

Residual connections for stable learning

In this project, we’ll build a proper multi-layer TCN from scratch and train it on a sequence prediction task where the future depends on past input patterns.

About:
✅ What It Does:
Constructs a stacked TCN with residual blocks

Uses dilated convolutions to capture both short- and long-range patterns

Trains on a custom rule-based synthetic sequence

Demonstrates how TCNs outperform RNNs in efficiency and performance for many tasks
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
 
# 1. Simulate data: next value depends on a sum of past 3 points
np.random.seed(42)
seq_len = 30
n = 1000
x = np.random.randn(n)
y = np.array([x[i-3] + x[i-1] for i in range(3, n)])  # dependency on t-1 and t-3
x = x[3:]  # align input
 
# Prepare sequences
def create_sequences(data, targets, seq_len):
    X, Y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i+seq_len])
        Y.append(targets[i+seq_len])
    return torch.FloatTensor(X).unsqueeze(1), torch.FloatTensor(Y)
 
X, Y = create_sequences(x, y, seq_len=seq_len)
dataset = TensorDataset(X, Y)
loader = DataLoader(dataset, batch_size=32, shuffle=True)
 
# 2. Define TCN Block with residual connection
class TCNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dilation):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size, 
                      padding=(kernel_size-1)*dilation, dilation=dilation),
            nn.ReLU(),
            nn.Conv1d(out_ch, out_ch, kernel_size, 
                      padding=(kernel_size-1)*dilation, dilation=dilation),
            nn.ReLU()
        )
        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None
 
    def forward(self, x):
        out = self.conv(x)
        res = x if self.downsample is None else self.downsample(x)
        return out[:, :, :-out.size(2) + res.size(2)] + res
 
# 3. Build TCN model
class TCN(nn.Module):
    def __init__(self, input_channels=1, levels=[16, 32, 64], kernel_size=3):
        super().__init__()
        layers = []
        for i, out_ch in enumerate(levels):
            dilation = 2 ** i
            in_ch = input_channels if i == 0 else levels[i-1]
            layers.append(TCNBlock(in_ch, out_ch, kernel_size, dilation))
        self.network = nn.Sequential(*layers)
        self.fc = nn.Linear(levels[-1], 1)
 
    def forward(self, x):
        x = self.network(x)
        return self.fc(x[:, :, -1]).squeeze()
 
model = TCN()
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
 
# 4. Train
for epoch in range(20):
    for batch_x, batch_y in loader:
        preds = model(batch_x)
        loss = loss_fn(preds, batch_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item():.6f}")
 
# 5. Evaluate
model.eval()
with torch.no_grad():
    preds = model(X).numpy()
 
plt.figure(figsize=(10, 4))
plt.plot(Y.numpy(), label="True")
plt.plot(preds, label="TCN Prediction", alpha=0.7)
plt.title("Temporal Convolutional Network – Sequence Forecasting")
plt.legend()
plt.grid(True)
plt.show()