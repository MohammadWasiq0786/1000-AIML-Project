"""
Project 281. ARIMA for stock price prediction
Description:
ARIMA (AutoRegressive Integrated Moving Average) is a classic time series forecasting technique, ideal for stationary datasets. It combines:

AR: dependency on past values

I: differencing to remove trends

MA: dependency on past forecast errors

We’ll use ARIMA to forecast future stock prices based on historical closing data.

About:
✅ What It Does:
Downloads real stock data using yfinance

Fits an ARIMA model to historical closing prices

Forecasts the next 30 business days

Visualizes both historical data and the forecasted trend
"""

import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
import yfinance as yf
 
# Load historical stock prices (e.g., Apple)
df = yf.download("AAPL", start="2020-01-01", end="2023-01-01")
data = df["Close"]
 
# Plot raw data
plt.figure(figsize=(10, 4))
plt.plot(data)
plt.title("Apple Stock Price")
plt.ylabel("Price ($)")
plt.grid(True)
plt.show()
 
# Fit ARIMA model (p=5, d=1, q=2)
model = ARIMA(data, order=(5, 1, 2))
model_fit = model.fit()
 
# Forecast next 30 days
forecast = model_fit.forecast(steps=30)
 
# Plot forecast
plt.figure(figsize=(10, 4))
plt.plot(data, label="Historical")
plt.plot(pd.date_range(data.index[-1], periods=30, freq='B'), forecast, label="Forecast")
plt.title("ARIMA Forecast – AAPL")
plt.ylabel("Price ($)")
plt.legend()
plt.grid(True)
plt.show()