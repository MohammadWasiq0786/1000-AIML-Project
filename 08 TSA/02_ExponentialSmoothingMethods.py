"""
Project 282. Exponential smoothing methods
Description:
Exponential Smoothing techniques are powerful for time series forecasting, especially when dealing with trend and seasonality. These models weigh recent observations more heavily, allowing quick adaptation to new patterns.

We’ll use Holt-Winters Exponential Smoothing, which supports:

Level (current value)

Trend (direction of the data)

Seasonality (recurring patterns)

About:
✅ What It Does:
Uses monthly-averaged stock prices for forecasting

Fits a Holt’s linear trend model (with additive trend)

Forecasts 6 months ahead

Ideal for series with trend but no clear seasonality
"""

import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import yfinance as yf
 
# Load dataset: monthly airline passengers (or use stock for non-seasonal)
df = yf.download("MSFT", start="2020-01-01", end="2023-01-01")
data = df["Close"].resample("M").mean()  # monthly average
 
# Plot original data
plt.figure(figsize=(10, 4))
plt.plot(data, label="Monthly Avg Close")
plt.title("Microsoft Monthly Stock Price")
plt.ylabel("Price ($)")
plt.grid(True)
plt.show()
 
# Apply Holt-Winters: additive trend, no seasonality
model = ExponentialSmoothing(data, trend='add', seasonal=None)
model_fit = model.fit()
 
# Forecast 6 future points
forecast = model_fit.forecast(steps=6)
 
# Plot results
plt.figure(figsize=(10, 4))
plt.plot(data, label="Observed")
plt.plot(forecast, label="Forecast", linestyle='--')
plt.title("Holt's Linear Trend Forecast – MSFT")
plt.ylabel("Price ($)")
plt.legend()
plt.grid(True)
plt.show()