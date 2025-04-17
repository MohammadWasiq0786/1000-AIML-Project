"""
Project 513: Commodity Price Forecasting
Description:
Commodity price forecasting involves predicting the future price of commodities like oil, gold, wheat, or natural gas based on historical data, market trends, and economic factors. In this project, we will predict the price of a commodity (e.g., gold) using time series forecasting methods like ARIMA or LSTM for more advanced predictions.

For real-world applications:

Use historical commodity price data from sources like Yahoo Finance, FRED, or Quandl.

Enhance the model with external factors like currency exchange rates, geopolitical events, or interest rates.

About:
✅ What It Does:
Downloads historical commodity price data (e.g., gold price) from Yahoo Finance.

Splits the data into training and test sets (80% for training, 20% for testing).

Fits an ARIMA model on the training data to model the commodity price time series.

Forecasts future prices on the test set using the trained ARIMA model.

Evaluates the model's accuracy using Mean Absolute Error (MAE).

Visualizes the actual and predicted commodity prices.

Key Extensions and Customizations:
Use multiple commodities: You can extend this project by forecasting prices for multiple commodities (e.g., oil, natural gas, or wheat).

Incorporate external data: Add external factors such as currency exchange rates, inflation rates, or geopolitical events that may impact commodity prices.

Advanced models: Use more advanced models like LSTM (Long Short-Term Memory) or Prophet for better forecasting on complex time series data.

Backtesting: Implement backtesting to simulate real trading conditions using historical data.
"""

import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
 
# 1. Download historical commodity price data (e.g., Gold price)
commodity_data = yf.download("GC=F", start="2015-01-01", end="2021-01-01")['Close']
 
# 2. Plot the historical commodity prices
plt.figure(figsize=(10, 6))
plt.plot(commodity_data)
plt.title('Gold Price (Commodity) - Historical Data')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.grid(True)
plt.show()
 
# 3. Split data into train and test sets
train_size = int(len(commodity_data) * 0.8)
train_data, test_data = commodity_data[:train_size], commodity_data[train_size:]
 
# 4. Fit ARIMA model (we choose p=5, d=1, q=0 arbitrarily for simplicity)
model = ARIMA(train_data, order=(5, 1, 0))
model_fit = model.fit()
 
# 5. Forecast commodity prices on test data
forecast_steps = len(test_data)
forecast = model_fit.forecast(steps=forecast_steps)
 
# 6. Plot the forecasted results vs actual data
plt.figure(figsize=(10, 6))
plt.plot(train_data, label='Training Data', color='blue')
plt.plot(test_data, label='Test Data', color='orange')
plt.plot(test_data.index, forecast, label='Predicted Data', color='green')
plt.title('Commodity Price Forecasting (Gold Price using ARIMA)')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.grid(True)
plt.show()
 
# 7. Evaluate the model (e.g., using Mean Absolute Error)
from sklearn.metrics import mean_absolute_error
 
mae = mean_absolute_error(test_data, forecast)
print(f'Mean Absolute Error for Commodity Price Forecasting: ${mae:.2f}')