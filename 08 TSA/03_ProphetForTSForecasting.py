"""
Project 283. Prophet for time series forecasting
Description:
Prophet, developed by Meta (Facebook), is a robust and easy-to-use time series forecasting tool. It's ideal for data with:

Trend

Seasonality

Holiday effects

Prophet automatically handles missing data, outliers, and daily/weekly/yearly patterns — perfect for business forecasting tasks.

About:
✅ What It Does:
Fetches historical stock prices

Fits a Prophet model on closing price

Forecasts 60 future days

Visualizes trends, seasonality, and forecast confidence intervals

Great for business-friendly forecasting
"""

import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
import yfinance as yf
 
# Download stock data (e.g., Google)
df = yf.download("GOOG", start="2020-01-01", end="2023-01-01")
data = df.reset_index()[["Date", "Close"]]
data.columns = ["ds", "y"]  # Prophet expects columns: ds (date), y (value)
 
# Initialize and fit Prophet model
model = Prophet()
model.fit(data)
 
# Create future dataframe (next 60 days)
future = model.make_future_dataframe(periods=60)
forecast = model.predict(future)
 
# Plot forecast
fig1 = model.plot(forecast)
plt.title("Prophet Forecast – Google Stock")
plt.ylabel("Price ($)")
plt.xlabel("Date")
plt.grid(True)
plt.show()
 
# Optional: plot trend + seasonality components
fig2 = model.plot_components(forecast)
plt.show()