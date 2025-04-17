"""
Project 481: Stock Price Prediction Model
Description:
Stock price prediction is a complex task that relies on historical data to forecast future prices. In this project, we will use LSTM (Long Short-Term Memory) networks, a type of recurrent neural network (RNN), to predict future stock prices based on previous price data.

We'll train the model on historical closing prices and use it to predict the next day's closing price.

✅ What It Does:
Downloads historical stock data (Apple in this case) using yfinance.

Normalizes the data with MinMaxScaler to scale the data to a range of [0, 1].

LSTM-based model predicts the next day's closing price based on the past 60 days' data.

Plots actual vs predicted stock prices for evaluation.

Key Extensions and Customizations:
Fine-tuning the model: You can adjust the number of LSTM units, dropout rate, or epochs for better results.

Use more features: You can add more features like technical indicators (Moving Average, RSI, etc.) to improve the model's accuracy.

Real-time predictions: Implement real-time stock price prediction using an API like Alpha Vantage for live data.

You can later use real stock data from:

Yahoo Finance API (via yfinance)

Alpha Vantage API

Quandl
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import yfinance as yf
 
# 1. Download stock data (e.g., Apple stock)
stock_data = yf.download("AAPL", start="2015-01-01", end="2021-01-01")
 
# 2. Preprocess the data (use closing prices)
close_prices = stock_data['Close'].values.reshape(-1, 1)
 
# 3. Scale the data (normalize values to the range [0,1])
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(close_prices)
 
# 4. Create data sequences (use 60 days' worth of data to predict the 61st day)
def create_dataset(data, time_step=60):
    X, y = [], []
    for i in range(time_step, len(data)):
        X.append(data[i-time_step:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)
 
X, y = create_dataset(scaled_data)
 
# 5. Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
 
# 6. Reshape the input data to match LSTM input format (samples, time steps, features)
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
 
# 7. Build the LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(units=1))
 
# 8. Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')
 
# 9. Train the model
history = model.fit(X_train, y_train, epochs=10, batch_size=32)
 
# 10. Predict stock prices
predictions = model.predict(X_test)
 
# 11. Inverse scale the predictions
predicted_stock_price = scaler.inverse_transform(predictions)
actual_stock_price = scaler.inverse_transform(y_test.reshape(-1, 1))
 
# 12. Plot the results
plt.figure(figsize=(14, 7))
plt.plot(actual_stock_price, color='blue', label='Actual Stock Price')
plt.plot(predicted_stock_price, color='red', label='Predicted Stock Price')
plt.title('Stock Price Prediction (LSTM)')
plt.xlabel('Time')
plt.ylabel('Stock Price (USD)')
plt.legend()
plt.show()