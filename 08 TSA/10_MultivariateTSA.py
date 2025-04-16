"""
Project 290. Multivariate time series analysis
Description:
Multivariate time series involve multiple dependent variables tracked over time — like temperature, humidity, and wind speed, or stock prices and trading volume. Analyzing them together helps capture cross-variable dependencies and interactions.

We’ll analyze stock closing price and volume together using correlation visualization and fit a Vector AutoRegression (VAR) model.

About:
✅ What It Does:
Downloads two related series: stock price + volume

Plots them together to inspect trends and correlation

Prepares data (via differencing) for VAR model

Forecasts multiple series together to capture interactions
"""


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from statsmodels.tsa.api import VAR
 
# Download stock data: Google
df = yf.download("GOOG", start="2020-01-01", end="2023-01-01")[["Close", "Volume"]]
 
# Normalize (optional for VAR)
data = df.copy()
data["Volume"] = data["Volume"] / 1e6  # scale volume in millions
 
# Visualize correlation
sns.heatmap(data.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Matrix: GOOG Time Series")
plt.show()
 
# Plot both series
data.plot(subplots=True, figsize=(10, 6), title=["Close Price", "Volume (Millions)"])
plt.suptitle("Google Stock – Multivariate Time Series", fontsize=14)
plt.tight_layout()
plt.grid(True)
plt.show()
 
# Prepare for VAR model
data = data.diff().dropna()  # stationarity
model = VAR(data)
results = model.fit(maxlags=15, ic='aic')
 
# Forecast next 5 days
forecast = results.forecast(data.values[-results.k_ar:], steps=5)
forecast_df = pd.DataFrame(forecast, columns=data.columns)
 
# Plot forecast
forecast_df.plot(title="Multivariate Forecast (VAR) – Next 5 Days", figsize=(10, 4))
plt.grid(True)
plt.show()