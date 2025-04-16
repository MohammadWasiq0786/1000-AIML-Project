"""
Project 289. Seasonality decomposition
Description:
Seasonality decomposition separates a time series into three components:

Trend: Long-term progression

Seasonality: Repeating short-term cycle

Residual: Noise or irregularities

This helps in understanding underlying patterns and preparing data for forecasting. We'll use seasonal decomposition of time series (STL) from statsmodels.

About:
✅ What It Does:
Resamples daily data to monthly averages for smoother seasonal patterns

Applies STL decomposition to extract:

Trend component

Seasonal component

Residual/noise component

Visualizes each to reveal hidden dynamics
"""


import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from statsmodels.tsa.seasonal import STL
 
# Load monthly stock data (resampled): Apple
df = yf.download("AAPL", start="2018-01-01", end="2023-01-01")
monthly = df["Close"].resample('M').mean()
 
# Apply STL decomposition
stl = STL(monthly, seasonal=13)
result = stl.fit()
 
# Plot components
result.plot()
plt.suptitle("STL Decomposition – Apple Monthly Stock Price", fontsize=14)
plt.tight_layout()
plt.grid(True)
plt.show()