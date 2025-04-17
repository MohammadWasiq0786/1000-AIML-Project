"""
Project 489: Time Series Forecasting for Finance
Description:
Time series forecasting for finance predicts future stock prices or market indices using historical data. In this project, we will use ARIMA (AutoRegressive Integrated Moving Average), a classic model for time series forecasting, to predict future stock prices based on historical closing prices.

For real-world use:

Use datasets like Yahoo Finance, Alpha Vantage, or Quandl.

You can also enhance this project by using more advanced models like LSTM or Prophet for better results.

✅ What It Does:
Downloads historical stock data (Apple in this case) and splits it into training and test sets.

Fits an ARIMA model on the training data and forecasts future stock prices for the test period.

Plots the stock prices: the actual historical prices, the forecasted values, and the model's prediction.

Evaluates the model using Mean Absolute Error (MAE).
"""

import numpy as np
import pandas as pd
import yfinance as yf
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
 
# 1. Download stock data (e.g., Apple stock)
stock_data = yf.download("AAPL", start="2015-01-01", end="2021-01-01")
stock_data = stock_data['Close']
 
# 2. Plot the stock price data
plt.figure(figsize=(10, 6))
plt.plot(stock_data)
plt.title('Apple Stock Price (2015 - 2021)')
plt.xlabel('Date')
plt.ylabel('Stock Price (USD)')
plt.show()
 
# 3. Split the data into train and test sets
train_size = int(len(stock_data) * 0.8)
train_data, test_data = stock_data[:train_size], stock_data[train_size:]
 
# 4. Fit the ARIMA model
# p = 5, d = 1, q = 0 are chosen arbitrarily; these can be optimized
model = ARIMA(train_data, order=(5, 1, 0))
model_fit = model.fit()
 
# 5. Make predictions
forecast_steps = len(test_data)
forecast = model_fit.forecast(steps=forecast_steps)
 
# 6. Plot the forecasted results
plt.figure(figsize=(10, 6))
plt.plot(train_data, label='Training Data', color='blue')
plt.plot(test_data, label='Test Data', color='orange')
plt.plot(test_data.index, forecast, label='Predicted Data', color='green')
plt.title('Stock Price Prediction (ARIMA)')
plt.xlabel('Date')
plt.ylabel('Stock Price (USD)')
plt.legend()
plt.show()
 
# 7. Evaluate the model (e.g., using Mean Absolute Error)
from sklearn.metrics import mean_absolute_error
 
mae = mean_absolute_error(test_data, forecast)
print(f'Mean Absolute Error: {mae:.2f}')