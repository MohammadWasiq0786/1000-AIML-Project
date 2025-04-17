"""
Project 491: Cryptocurrency Price Prediction
Description:
Cryptocurrency price prediction aims to forecast future prices of cryptocurrencies, such as Bitcoin, Ethereum, or Litecoin, based on historical market data. In this project, we will use a LSTM (Long Short-Term Memory) model to predict the price of Bitcoin based on historical data, similar to stock price prediction, but for volatile cryptocurrencies.

For real-world use:

You can use APIs like Binance, CoinGecko, or CryptoCompare for real-time cryptocurrency data.

To improve accuracy, use additional features such as social media sentiment, technical indicators, or market news.

✅ What It Does:
Downloads historical Bitcoin prices using yfinance.

Preprocesses the data using MinMaxScaler to normalize it for LSTM.

Trains an LSTM model to predict Bitcoin's price based on the past 60 days.

Plots actual vs. predicted prices to evaluate the model.

Key Extensions and Customizations:
Use multiple cryptocurrencies: Extend the model to predict Ethereum, Litecoin, or any other cryptocurrency by adding them as additional features.

Use technical indicators: Add moving averages, RSI, or MACD as additional features for more accurate predictions.

Social media sentiment: Integrate data from Twitter or Reddit to predict market sentiment and improve forecasting.

Deploy for real-time predictions: Use WebSocket connections to stream live data and make real-time predictions.
"""

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt
 
# 1. Download cryptocurrency data (Bitcoin price)
crypto_data = yf.download("BTC-USD", start="2015-01-01", end="2021-01-01")
crypto_data = crypto_data['Close']
 
# 2. Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(crypto_data.values.reshape(-1, 1))
 
# 3. Create dataset with time steps
def create_dataset(data, time_step=60):
    X, y = [], []
    for i in range(time_step, len(data)):
        X.append(data[i-time_step:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)
 
X, y = create_dataset(scaled_data)
 
# 4. Train/test split
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]
 
# 5. Reshape the input data for LSTM
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
 
# 6. Build the LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(units=1))
 
# 7. Compile and train the model
model.compile(optimizer='adam', loss='mean_squared_error')
history = model.fit(X_train, y_train, epochs=10, batch_size=32)
 
# 8. Make predictions
predictions = model.predict(X_test)
 
# 9. Inverse scaling for predictions
predicted_prices = scaler.inverse_transform(predictions)
actual_prices = scaler.inverse_transform(y_test.reshape(-1, 1))
 
# 10. Plot the results
plt.figure(figsize=(14, 7))
plt.plot(actual_prices, label='Actual Bitcoin Price', color='blue')
plt.plot(predicted_prices, label='Predicted Bitcoin Price', color='red')
plt.title('Bitcoin Price Prediction with LSTM')
plt.xlabel('Time')
plt.ylabel('Price (USD)')
plt.legend()
plt.show()