"""
Project 473. Medical time series analysis
Description:
Medical time series analysis involves processing sequential data like ECG, ICU vitals, or blood pressure over time to extract trends, detect anomalies, and predict outcomes. In this project, we simulate vital sign data and use a Recurrent Neural Network (RNN) to predict a patient's future heart rate.

✅ What It Does:
Simulates heart rate trends with natural variability.

Builds an LSTM model to predict the next value in the time series.

Can be extended to:

Predict multi-vital signals: BP, SpO2, respiration rate

Detect critical events like arrhythmia or sepsis

Integrate with real ICU data dashboards

Real-world data sources:

MIMIC-III / MIMIC-IV

PhysioNet (e.g., ICU Waveform Database, ECG datasets)
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
 
# 1. Simulate patient heart rate time series data
np.random.seed(42)
heart_rate = np.random.normal(75, 5, 300) + np.sin(np.linspace(0, 20, 300)) * 10
 
# 2. Create sequences for supervised learning
class HeartRateDataset(Dataset):
    def __init__(self, series, seq_len=10):
        self.X = []
        self.y = []
        for i in range(len(series) - seq_len):
            self.X.append(series[i:i+seq_len])
            self.y.append(series[i+seq_len])
        self.X = torch.tensor(self.X, dtype=torch.float32).unsqueeze(-1)
        self.y = torch.tensor(self.y, dtype=torch.float32)
 
    def __len__(self):
        return len(self.X)
 
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
 
dataset = HeartRateDataset(heart_rate)
loader = DataLoader(dataset, batch_size=16, shuffle=True)
 
# 3. Define LSTM model
class HeartRateLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=32, batch_first=True)
        self.fc = nn.Linear(32, 1)
 
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :]).squeeze()
 
model = HeartRateLSTM()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()
 
# 4. Training loop
for epoch in range(1, 6):
    model.train()
    total_loss = 0
    for X_batch, y_batch in loader:
        preds = model(X_batch)
        loss = loss_fn(preds, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch} - Loss: {total_loss / len(loader):.4f}")