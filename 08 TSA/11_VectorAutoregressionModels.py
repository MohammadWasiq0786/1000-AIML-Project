"""
Project 291. Vector autoregression models
Description:
Vector Autoregression (VAR) is a generalization of AR models to multivariate time series, where each variable depends not only on its own past values but also on the past values of other variables in the system.

In this project, we’ll fit a VAR model on multiple stock metrics (e.g., Close price, Volume), use it to forecast future values, and analyze lag relationships.

About:
✅ What It Does:
Loads Meta stock data (close + volume)

Applies differencing to make the series stationary

Fits a VAR model, selects optimal lag with AIC

Forecasts both variables together, capturing mutual influence

Plots predictions vs actuals for visual inspection
"""
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from statsmodels.tsa.api import VAR
 
# Load multiple time series: Meta stock data
df = yf.download("META", start="2020-01-01", end="2023-01-01")[["Close", "Volume"]]
 
# Normalize volume (for comparison) and preprocess
df["Volume"] = df["Volume"] / 1e6  # scale volume in millions
df = df.dropna()
 
# Convert to stationary series (first difference)
df_diff = df.diff().dropna()
 
# Split into training and testing sets
n_obs = 10  # reserve last 10 days for forecasting
train, test = df_diff[:-n_obs], df_diff[-n_obs:]
 
# Fit VAR model
model = VAR(train)
results = model.fit(maxlags=15, ic='aic')
print(f"Selected Lag Order: {results.k_ar}")
 
# Forecast
forecast_input = train.values[-results.k_ar:]
forecast = results.forecast(y=forecast_input, steps=n_obs)
forecast_df = pd.DataFrame(forecast, index=test.index, columns=train.columns)
 
# Plot forecasts vs actual
for col in train.columns:
    plt.figure(figsize=(8, 3))
    plt.plot(test.index, test[col], label="Actual")
    plt.plot(forecast_df.index, forecast_df[col], label="Forecast", linestyle="--")
    plt.title(f"VAR Forecast vs Actual – {col}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()