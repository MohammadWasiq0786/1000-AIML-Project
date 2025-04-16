"""
Project 310. Self-supervised learning for time series
Description:
Self-Supervised Learning (SSL) trains models using pretext tasks derived from unlabeled data. The model learns meaningful representations that can later be fine-tuned for downstream tasks (e.g., classification or forecasting).

For time series, common SSL tasks include:

Temporal contrastive learning

Masked value prediction

Forecasting next steps

In this project, we’ll train a model to reconstruct masked values in a time series (similar to BERT for text), teaching it to understand temporal patterns.

About:
✅ What It Does:
Trains a model to predict masked time steps using only unlabeled data

Learns useful representations via a self-supervised objective

Visualizes prediction accuracy on masked regions

Can be fine-tuned later for downstream tasks (classification, forecasting, etc.)
"""

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
 
# 1. Generate a clean sine wave time series
np.random.seed(42)
t = np.linspace(0, 50, 1000)
series = np.sin(t) + 0.05 * np.random.randn(len(t))
 
# 2. Create masked sequences for SSL
def create_masked_dataset(data, seq_len=50, mask_prob=0.2):
    X, Y, M = [], [], []
    for i in range(len(data) - seq_len):
        seq = data[i:i+seq_len]
        mask = np.random.rand(seq_len) < mask_prob
        masked_seq = seq.copy()
        masked_seq[mask] = 0.0
        X.append(masked_seq)
        Y.append(seq)
        M.append(mask.astype(float))
    return torch.FloatTensor(X).unsqueeze(-1), torch.FloatTensor(Y).unsqueeze(-1), torch.FloatTensor(M).unsqueeze(-1)
 
seq_len = 50
X, Y, mask = create_masked_dataset(series, seq_len)
dataset = TensorDataset(X, Y, mask)
loader = DataLoader(dataset, batch_size=32, shuffle=True)
 
# 3. Define SSL model (CNN Encoder + Reconstructor)
class SSLTimeSeriesModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 32, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.decoder = nn.Conv1d(32, 1, kernel_size=1)
 
    def forward(self, x):
        x = x.transpose(1, 2)  # (B, 1, T) → (B, C, T)
        z = self.encoder(x)
        out = self.decoder(z).transpose(1, 2)
        return out
 
model = SSLTimeSeriesModel()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss(reduction='none')  # we'll mask the loss manually
 
# 4. Train model to reconstruct masked values
for epoch in range(10):
    for xb, yb, mb in loader:
        pred = model(xb)
        loss = loss_fn(pred, yb)
        masked_loss = (loss * mb).sum() / mb.sum()
        optimizer.zero_grad()
        masked_loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}, Masked Loss: {masked_loss.item():.5f}")
 
# 5. Evaluate on one sequence
model.eval()
with torch.no_grad():
    example_x = X[100].unsqueeze(0)
    example_y = Y[100].squeeze().numpy()
    example_mask = mask[100].squeeze().numpy()
    pred_y = model(example_x).squeeze().numpy()
 
plt.figure(figsize=(10, 4))
plt.plot(example_y, label="Original")
plt.plot(pred_y, label="Predicted")
plt.plot(np.where(example_mask > 0)[0], example_y[example_mask > 0], "ro", label="Masked Targets")
plt.title("Self-Supervised Learning – Masked Time Series Reconstruction")
plt.legend()
plt.grid(True)
plt.show()