"""
Project 285. GRU for sequence modeling
Description:
GRU (Gated Recurrent Unit) is a simplified version of LSTM that retains long-term memory with fewer gates, making it faster and more efficient. GRUs work well in time series and sequential data, especially when computation or training time is a constraint.

In this project, we’ll use a GRU to model and predict the next day’s stock price based on the last 30 days of prices.

About:
✅ What It Does:
Uses a GRU-based RNN to predict stock prices

Trains on normalized sequences of 30-day windows

Compares predicted vs actual prices

Faster than LSTM with comparable accuracy
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
 
# 1. Load stock data
df = yf.download("NFLX", start="2020-01-01", end="2023-01-01")
data = df["Close"].values.reshape(-1, 1)
 
# 2. Normalize and create sequences
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)
 
def create_sequences(data, seq_len=30):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len])
    return np.array(X), np.array(y)
 
X, y = create_sequences(scaled_data)
X_train, y_train = torch.FloatTensor(X), torch.FloatTensor(y)
 
# 3. Define GRU Model
class GRUModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=50):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
 
    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])
 
model = GRUModel()
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
 
# 4. Train the model
epochs = 20
for epoch in range(epochs):
    model.train()
    output = model(X_train)
    loss = loss_fn(output, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item():.6f}")
 
# 5. Predict and plot results
model.eval()
predicted = model(X_train).detach().numpy()
predicted_prices = scaler.inverse_transform(predicted)
actual_prices = scaler.inverse_transform(y_train)
 
plt.figure(figsize=(10, 4))
plt.plot(actual_prices, label="Actual")
plt.plot(predicted_prices, label="Predicted")
plt.title("GRU Forecast – Netflix Stock")
plt.xlabel("Time")
plt.ylabel("Price ($)")
plt.legend()
plt.grid(True)
plt.show()