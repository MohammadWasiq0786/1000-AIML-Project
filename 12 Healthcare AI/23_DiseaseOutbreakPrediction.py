"""
Project 463. Disease outbreak prediction
Description:
Disease outbreak prediction models aim to forecast future occurrences or spread of infectious diseases like flu, COVID-19, or dengue using historical case data, environmental factors, or mobility patterns. In this project, we simulate time-series data and train a basic LSTM model to predict future case counts.

✅ What It Does:
Simulates historical disease case counts over time.

Trains an LSTM model to forecast future infection levels.

Extendable to:

Add climate variables (temperature, humidity)

Combine with geospatial data for regional outbreak forecasting

Deploy as an early warning dashboard

For real-world datasets:

FluNet, COVID-19 time-series (Johns Hopkins)

WHO, CDC, or Open Disease Data
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
 
# 1. Simulated weekly case count (e.g., for flu)
np.random.seed(42)
cases = np.random.poisson(lam=100, size=100) + np.linspace(0, 50, 100).astype(int)
 
# 2. Create sequences for LSTM
class OutbreakDataset(Dataset):
    def __init__(self, data, seq_len=10):
        self.X = []
        self.y = []
        for i in range(len(data) - seq_len):
            self.X.append(data[i:i+seq_len])
            self.y.append(data[i+seq_len])
        self.X = torch.tensor(self.X, dtype=torch.float32).unsqueeze(-1)
        self.y = torch.tensor(self.y, dtype=torch.float32)
 
    def __len__(self):
        return len(self.X)
 
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
 
dataset = OutbreakDataset(cases)
loader = DataLoader(dataset, batch_size=8, shuffle=True)
 
# 3. LSTM model
class LSTMOutbreakPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=32, batch_first=True)
        self.fc = nn.Linear(32, 1)
 
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :]).squeeze()
 
model = LSTMOutbreakPredictor()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()
 
# 4. Training loop
for epoch in range(1, 6):
    model.train()
    total_loss = 0
    for seqs, targets in loader:
        optimizer.zero_grad()
        preds = model(seqs)
        loss = loss_fn(preds, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch}, Loss: {total_loss / len(loader):.4f}")