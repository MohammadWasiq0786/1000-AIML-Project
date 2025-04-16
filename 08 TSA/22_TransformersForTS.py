"""
Project 302. Transformers for time series
Description:
Transformers, originally designed for NLP, have revolutionized time series modeling by replacing recurrence with self-attention, enabling:

Parallel processing of sequences

Better handling of long-range dependencies

Interpretability through attention weights

In this project, we’ll build a simplified Transformer encoder to forecast the next value in a time series.

About:
✅ What It Does:
Builds a simple Transformer encoder with self-attention

Uses only the last encoded time step for prediction

Handles sequence dependencies without recurrence

Forecasts future values from a noisy sine wave
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
 
# 1. Create synthetic sine wave data
np.random.seed(42)
t = np.linspace(0, 100, 1000)
series = np.sin(0.1 * t) + 0.1 * np.random.randn(1000)
 
# 2. Prepare data as sequences
def create_sequences(data, seq_len=50):
    X, Y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i+seq_len])
        Y.append(data[i+seq_len])
    return torch.FloatTensor(X).unsqueeze(-1), torch.FloatTensor(Y)
 
seq_len = 50
X, Y = create_sequences(series, seq_len)
dataset = TensorDataset(X, Y)
loader = DataLoader(dataset, batch_size=32, shuffle=True)
 
# 3. Transformer-based model
class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_dim=1, d_model=64, nhead=4, num_layers=2):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(d_model, 1)
 
    def forward(self, x):
        x = self.input_proj(x)                        # [B, T, d_model]
        x = self.transformer(x)                       # [B, T, d_model]
        return self.output_proj(x[:, -1, :]).squeeze()  # Predict using last time step
 
model = TimeSeriesTransformer()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()
 
# 4. Training loop
for epoch in range(20):
    for batch_x, batch_y in loader:
        preds = model(batch_x)
        loss = loss_fn(preds, batch_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item():.6f}")
 
# 5. Evaluation
model.eval()
with torch.no_grad():
    preds = model(X).numpy()
 
plt.figure(figsize=(10, 4))
plt.plot(Y.numpy(), label="True")
plt.plot(preds, label="Transformer Prediction", alpha=0.7)
plt.title("Transformer – Time Series Forecasting")
plt.legend()
plt.grid(True)
plt.show()