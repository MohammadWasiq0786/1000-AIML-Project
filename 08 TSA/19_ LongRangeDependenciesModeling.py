"""
Project 299. Long-range dependencies modeling
Description:
Many time series models struggle with capturing long-range dependencies — patterns or correlations that span across long time horizons. To model this effectively, we use architectures like:

Dilated RNNs

Attention-based models

Temporal Convolutional Networks (TCNs)

In this project, we’ll implement a simple Temporal Convolutional Network (TCN) to predict values that depend on a long past window.

About:
✅ What It Does:
Creates a synthetic signal where the output depends on inputs 30 and 60 steps ago

Builds a TCN to handle long-term dependencies

Learns using dilated convolutions with causal structure

Plots prediction vs actual output
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
 
# 1. Simulate a sequence with long-range dependency
# Output depends on x[t-30] + x[t-60]
np.random.seed(42)
n = 1000
x = np.random.randn(n)
y = np.zeros_like(x)
 
for t in range(60, n):
    y[t] = x[t - 30] + x[t - 60] + np.random.normal(scale=0.1)
 
# 2. Prepare sequences for TCN input
def create_sequences(x, y, seq_len=60):
    X, Y = [], []
    for i in range(seq_len, len(x)):
        X.append(x[i-seq_len:i])
        Y.append(y[i])
    return torch.FloatTensor(X).unsqueeze(1), torch.FloatTensor(Y)
 
X, Y = create_sequences(x, y)
dataset = TensorDataset(X, Y)
loader = DataLoader(dataset, batch_size=32, shuffle=True)
 
# 3. Define a simple TCN model
class TemporalConvNet(nn.Module):
    def __init__(self, input_channels=1, num_channels=[16, 32], kernel_size=3):
        super().__init__()
        layers = []
        for i in range(len(num_channels)):
            dilation_size = 2 ** i
            in_channels = input_channels if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [
                nn.Conv1d(in_channels, out_channels, kernel_size,
                          padding=(kernel_size - 1) * dilation_size,
                          dilation=dilation_size),
                nn.ReLU()
            ]
        self.network = nn.Sequential(*layers)
        self.output_layer = nn.Linear(num_channels[-1], 1)
 
    def forward(self, x):
        x = self.network(x)
        x = x[:, :, -1]  # last time step
        return self.output_layer(x).squeeze()
 
model = TemporalConvNet()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()
 
# 4. Train the model
for epoch in range(20):
    for batch_x, batch_y in loader:
        preds = model(batch_x)
        loss = loss_fn(preds, batch_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item():.5f}")
 
# 5. Evaluate and plot
model.eval()
with torch.no_grad():
    predictions = model(X).numpy()
 
plt.figure(figsize=(10, 4))
plt.plot(y[60:], label="True")
plt.plot(predictions, label="Predicted", alpha=0.7)
plt.title("Modeling Long-Range Dependencies with TCN")
plt.legend()
plt.grid(True)
plt.show()