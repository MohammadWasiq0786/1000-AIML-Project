"""
Project 314. Quantile regression for time series
Description:
Quantile Regression estimates conditional quantiles (like the 10th, 50th, or 90th percentile) instead of the mean. In time series, this gives:

Prediction intervals without assuming Gaussian error

Robust estimates against outliers

A way to model asymmetric uncertainty

We’ll build a neural network to predict multiple quantiles for each future value in a sequence.

About:
✅ What It Does:
Trains a model to predict the 10th, 50th, and 90th percentile of the next value

Uses quantile loss to handle asymmetric errors

Plots prediction intervals to visualize uncertainty

Helps in risk-aware forecasting (e.g., best/worst case scenarios)
"""

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
 
# 1. Generate noisy sine wave
np.random.seed(42)
t = np.linspace(0, 50, 1000)
series = np.sin(t) + 0.1 * np.random.randn(1000)
 
# 2. Create sequences
def create_sequences(data, seq_len=30):
    X, Y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i+seq_len])
        Y.append(data[i+seq_len])
    return torch.FloatTensor(X).unsqueeze(-1), torch.FloatTensor(Y)
 
X, Y = create_sequences(series, seq_len=30)
dataset = TensorDataset(X, Y)
loader = DataLoader(dataset, batch_size=32, shuffle=True)
 
# 3. Quantile loss function
def quantile_loss(preds, target, q):
    error = target - preds
    return torch.max(q * error, (q - 1) * error).mean()
 
# 4. Model to predict 3 quantiles (10%, 50%, 90%)
class QuantileRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(30, 64),
            nn.ReLU(),
            nn.Linear(64, 3)  # output for q10, q50, q90
        )
 
    def forward(self, x):
        return self.net(x)
 
model = QuantileRegressor()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
 
# 5. Train using composite quantile loss
quantiles = [0.1, 0.5, 0.9]
 
for epoch in range(10):
    for xb, yb in loader:
        preds = model(xb)
        loss = sum(quantile_loss(preds[:, i], yb, q) for i, q in enumerate(quantiles))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}, Composite Quantile Loss: {loss.item():.5f}")
 
# 6. Evaluate
model.eval()
with torch.no_grad():
    preds = model(X).numpy()
    true = Y.numpy()
 
# 7. Plot predictions with quantile intervals
plt.figure(figsize=(10, 4))
plt.plot(true, label="True", alpha=0.6)
plt.plot(preds[:, 1], label="Median (q50)", linewidth=2)
plt.fill_between(np.arange(len(preds)), preds[:, 0], preds[:, 2], color='orange', alpha=0.3, label="q10–q90 Interval")
plt.title("Quantile Regression – Time Series Forecasting")
plt.legend()
plt.grid(True)
plt.show()