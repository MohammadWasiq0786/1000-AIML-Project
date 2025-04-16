"""
Project 25. Time series forecasting with ARIMA
Description:
ARIMA (AutoRegressive Integrated Moving Average) is a classic statistical model used for time series forecasting. It combines autoregression (AR), differencing (I), and moving average (MA) to predict future values based on historical data. In this project, we use statsmodels to build and evaluate an ARIMA model on a sample time series dataset.

About:
📊 What This Project Shows:
Simulates a real-world trend with seasonal noise

Builds an ARIMA(1,1,1) model for forecasting

Compares predicted vs. actual values and evaluates using MSE
"""

# Install necessary library if not already installed
# pip install statsmodels
 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
 
# Generate a simple synthetic time series (e.g., monthly sales data)
np.random.seed(42)
months = pd.date_range(start="2022-01", periods=36, freq="M")
sales = np.linspace(200, 300, 36) + np.random.normal(scale=10, size=36)  # upward trend + noise
df = pd.DataFrame({"Date": months, "Sales": sales}).set_index("Date")
 
# Plot original data
df.plot(title="Monthly Sales Over Time", figsize=(10, 4))
plt.ylabel("Sales")
plt.grid(True)
plt.show()
 
# Split data into train and test
train = df.iloc[:-6]
test = df.iloc[-6:]
 
# Fit ARIMA model (order = (p,d,q))
model = ARIMA(train, order=(1,1,1))  # ARIMA(1,1,1) = AR + Differencing + MA
model_fit = model.fit()
 
# Forecast the next 6 periods (same as test set)
forecast = model_fit.forecast(steps=6)
forecast = pd.Series(forecast, index=test.index)
 
# Plot forecast vs actual
plt.figure(figsize=(10, 4))
plt.plot(train, label='Train')
plt.plot(test, label='Test', color='orange')
plt.plot(forecast, label='Forecast', color='green', linestyle='--')
plt.title("ARIMA Forecast vs Actual")
plt.ylabel("Sales")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
 
# Print forecast and evaluate
print("\nForecasted Values:\n", forecast)
print("\nMean Squared Error (MSE):", mean_squared_error(test, forecast))