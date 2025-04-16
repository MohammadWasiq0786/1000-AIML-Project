"""
Project 308. Transfer learning for time series
Description:
Transfer learning for time series allows a model trained on one dataset (source domain) to be reused or fine-tuned on another (target domain), saving training time and boosting performance — especially when the target data is scarce.

We’ll:

Train a model on one synthetic dataset

Fine-tune it on another with similar but slightly shifted dynamics

About:
✅ What It Does:
Learns shared patterns from source time series (sine wave)

Transfers knowledge to a similar task with shifted dynamics

Shows how fine-tuning accelerates convergence and improves results

Applicable in real-world when labeled data is limited in new domains
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
 
# 1. Generate two synthetic sine wave datasets (source and target)
def generate_series(shift, noise=0.1, n=500):
    t = np.linspace(0, 20, n)
    series = np.sin(t + shift) + noise * np.random.randn(n)
    return series
 
seq_len = 30
 
def create_sequences(data, seq_len):
    X, Y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i+seq_len])
        Y.append(data[i+seq_len])
    return torch.FloatTensor(X).unsqueeze(-1), torch.FloatTensor(Y)
 
# Source data: sine wave with no shift
source = generate_series(shift=0)
X_src, Y_src = create_sequences(source, seq_len)
src_loader = DataLoader(TensorDataset(X_src, Y_src), batch_size=32, shuffle=True)
 
# Target data: sine wave with phase shift
target = generate_series(shift=np.pi/4)
X_tgt, Y_tgt = create_sequences(target, seq_len)
tgt_loader = DataLoader(TensorDataset(X_tgt, Y_tgt), batch_size=32, shuffle=True)
 
# 2. Define LSTM model
class LSTMModel(nn.Module):
    def __init__(self, hidden_dim=32):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
 
    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        return self.fc(hn[-1]).squeeze()
 
model = LSTMModel()
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
loss_fn = nn.MSELoss()
 
# 3. Pretrain on source data
print("🔄 Pretraining on source domain...")
for epoch in range(10):
    for xb, yb in src_loader:
        preds = model(xb)
        loss = loss_fn(preds, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item():.5f}")
 
# 4. Fine-tune on target data
print("\n🎯 Fine-tuning on target domain...")
for epoch in range(5):
    for xb, yb in tgt_loader:
        preds = model(xb)
        loss = loss_fn(preds, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item():.5f}")
 
# 5. Predict on target data
model.eval()
with torch.no_grad():
    preds = model(X_tgt).numpy()
 
# 6. Plot predictions
plt.figure(figsize=(10, 4))
plt.plot(Y_tgt.numpy(), label="True Target")
plt.plot(preds, label="Predicted", alpha=0.7)
plt.title("Transfer Learning – LSTM on Shifted Sine Wave")
plt.legend()
plt.grid(True)
plt.show()