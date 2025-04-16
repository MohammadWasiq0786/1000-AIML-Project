"""
Project 315. Ensemble methods for time series
Description:
Ensemble methods combine predictions from multiple models to:

Improve forecast accuracy

Reduce variance and overfitting

Increase robustness across data variations

In this project, we’ll build an ensemble of:

ARIMA (for trend)

Prophet (for seasonality)

LSTM (for complex nonlinear patterns)

Then, we’ll average their predictions for a final forecast.

About:
✅ What It Does:
Uses three different models to forecast the next 30 days

Combines their strengths by averaging predictions

Visualizes individual and ensemble forecasts

Produces a more robust and accurate prediction
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
 
# 1. Simulate time series with trend + seasonality + noise
np.random.seed(42)
n = 150
t = pd.date_range(start="2020-01", periods=n, freq="D")
y = 0.05 * np.arange(n) + 3 * np.sin(2 * np.pi * np.arange(n) / 30) + np.random.normal(0, 0.5, n)
df = pd.DataFrame({'ds': t, 'y': y})
 
# 2. Forecast using ARIMA
arima_model = ARIMA(df['y'], order=(5, 1, 0)).fit()
arima_forecast = arima_model.forecast(steps=30).values
 
# 3. Forecast using Prophet
prophet_model = Prophet()
prophet_model.fit(df)
future = prophet_model.make_future_dataframe(periods=30)
prophet_forecast = prophet_model.predict(future)['yhat'][-30:].values
 
# 4. Forecast using LSTM
def create_sequences(data, seq_len=30):
    X, Y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i+seq_len])
        Y.append(data[i+seq_len])
    return torch.FloatTensor(X).unsqueeze(-1), torch.FloatTensor(Y)
 
y_scaled = (y - y.mean()) / y.std()
X, Y = create_sequences(y_scaled)
loader = DataLoader(TensorDataset(X, Y), batch_size=32, shuffle=True)
 
class LSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(1, 32, batch_first=True)
        self.fc = nn.Linear(32, 1)
 
    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return self.fc(h[-1])
 
lstm_model = LSTM()
optimizer = torch.optim.Adam(lstm_model.parameters(), lr=0.01)
loss_fn = nn.MSELoss()
 
# Train LSTM
for epoch in range(10):
    for xb, yb in loader:
        preds = lstm_model(xb).squeeze()
        loss = loss_fn(preds, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
 
# Forecast next 30 using last 30 points
lstm_input = torch.FloatTensor(y_scaled[-30:]).unsqueeze(0).unsqueeze(-1)
lstm_forecast = []
lstm_model.eval()
with torch.no_grad():
    for _ in range(30):
        pred = lstm_model(lstm_input).squeeze().item()
        lstm_forecast.append(pred)
        lstm_input = torch.cat([lstm_input[:, 1:], torch.tensor([[[pred]]])], dim=1)
lstm_forecast = np.array(lstm_forecast) * y.std() + y.mean()
 
# 5. Average predictions (ensemble)
ensemble_forecast = (arima_forecast + prophet_forecast + lstm_forecast) / 3
 
# 6. Plot all
plt.figure(figsize=(12, 5))
plt.plot(t, y, label="Observed")
forecast_dates = pd.date_range(t[-1] + pd.Timedelta(days=1), periods=30)
plt.plot(forecast_dates, arima_forecast, label="ARIMA")
plt.plot(forecast_dates, prophet_forecast, label="Prophet")
plt.plot(forecast_dates, lstm_forecast, label="LSTM")
plt.plot(forecast_dates, ensemble_forecast, label="Ensemble", linewidth=2, linestyle="--")
plt.title("Ensemble Time Series Forecasting (ARIMA + Prophet + LSTM)")
plt.legend()
plt.grid(True)
plt.show()