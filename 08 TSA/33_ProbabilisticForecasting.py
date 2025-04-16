"""
Project 313. Probabilistic forecasting
Description:
Unlike point forecasts, probabilistic forecasting predicts a distribution over future values, allowing us to capture:

Uncertainty

Prediction intervals

Risk-sensitive decisions

In this project, we’ll use Facebook’s Prophet to forecast a time series and extract confidence intervals (upper and lower bounds) for each prediction.

About:
✅ What It Does:
Builds a trend + seasonal model using Prophet

Produces a distribution over future values

Outputs yhat (mean) and yhat_lower/upper (intervals)

Plots forecast with uncertainty ribbons

This is ideal when point predictions aren’t enough — e.g., for inventory planning, risk management, or pricing.
"""

# Install if needed:
# pip install prophet
 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
 
# 1. Generate synthetic trend + seasonality data
np.random.seed(42)
periods = 100
t = pd.date_range(start="2020-01-01", periods=periods, freq='D')
trend = np.linspace(10, 30, periods)
seasonal = 5 * np.sin(2 * np.pi * t.dayofyear / 365)
noise = np.random.normal(0, 1.5, periods)
y = trend + seasonal + noise
 
df = pd.DataFrame({'ds': t, 'y': y})
 
# 2. Fit Prophet model
model = Prophet()
model.fit(df)
 
# 3. Create future dataframe and forecast
future = model.make_future_dataframe(periods=30)
forecast = model.predict(future)
 
# 4. Plot with uncertainty intervals
fig = model.plot(forecast)
plt.title("Probabilistic Forecasting with Prophet")
plt.grid(True)
plt.show()
 
# 5. Print sample forecast with intervals
print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())