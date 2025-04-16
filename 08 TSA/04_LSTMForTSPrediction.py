"""
Project 284. LSTM for time series prediction
Description:
LSTM (Long Short-Term Memory) networks are a type of recurrent neural network (RNN) designed to learn from sequential data and capture long-term dependencies. They're powerful for time series forecasting — especially when the relationship between past and future values is complex.

We’ll use an LSTM to predict the next stock price based on previous price sequences.

About:
✅ What It Does:
Uses 30 previous days to predict the next closing price

Scales data with MinMaxScaler

Builds and trains an LSTM-based neural network

Plots predictions vs actual values on training data
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
 
# 1. Load and preprocess data
df = yf.download("TSLA", start="2020-01-01", end="2023-01-01")
data = df["Close"].values.reshape(-1, 1)
 
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)
 
def create_sequences(data, seq_len=30):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len])
    return np.array(X), np.array(y)
 
seq_len = 30
X, y = create_sequences(scaled_data, seq_len)
X_train, y_train = torch.FloatTensor(X), torch.FloatTensor(y)
 
# 2. Define LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=50):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
 
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])
 
model = LSTMModel()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
 
# 3. Train model
epochs = 20
for epoch in range(epochs):
    model.train()
    output = model(X_train)
    loss = criterion(output, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.5f}")
 
# 4. Predict and plot
model.eval()
predicted = model(X_train).detach().numpy()
predicted_prices = scaler.inverse_transform(predicted)
true_prices = scaler.inverse_transform(y_train)
 
plt.figure(figsize=(10, 4))
plt.plot(true_prices, label="Actual")
plt.plot(predicted_prices, label="Predicted")
plt.title("LSTM Forecast – Tesla Stock")
plt.xlabel("Time")
plt.ylabel("Price ($)")
plt.legend()
plt.grid(True)
plt.show()