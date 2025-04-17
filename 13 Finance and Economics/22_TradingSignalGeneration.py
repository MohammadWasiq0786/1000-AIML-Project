"""
Project 502: Trading Signal Generation
Description:
Trading signal generation is an essential part of algorithmic trading, where signals indicate whether to buy, sell, or hold an asset. These signals are typically generated based on technical indicators, such as moving averages, RSI, MACD, or custom strategies. In this project, we will generate trading signals based on Moving Average Convergence Divergence (MACD) and Relative Strength Index (RSI).

For real-world applications:

Integrate with real-time market data from APIs like Alpaca, Binance, or Yahoo Finance.

Enhance the strategy by using other technical indicators or machine learning models for more robust trading signals.

✅ What It Does:
Downloads stock data using Yahoo Finance (Apple stock in this case).

Calculates MACD (Moving Average Convergence Divergence) and RSI (Relative Strength Index) indicators.

Generates trading signals:

Buy Signal: When MACD crosses above the signal line and RSI is below 30 (indicating the stock is oversold).

Sell Signal: When MACD crosses below the signal line and RSI is above 70 (indicating the stock is overbought).

Visualizes the stock price, buy/sell signals, and indicators on plots.

Key Extensions and Customizations:
Real-time data: Replace the static data with real-time stock prices from Alpaca, Binance, or Yahoo Finance.

Additional indicators: Incorporate other indicators like Bollinger Bands, Moving Average Crossovers, or Fibonacci retracements for better signal generation.

Backtesting: Implement a backtesting framework to evaluate the performance of this strategy over a historical dataset.

Machine learning integration: Combine these indicators with machine learning models to generate more robust predictions for trading signals.
"""

import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
 
# 1. Download historical stock data (e.g., Apple stock)
stock_data = yf.download("AAPL", start="2019-01-01", end="2021-01-01")
 
# 2. Calculate MACD (12-day EMA - 26-day EMA)
stock_data['EMA12'] = stock_data['Close'].ewm(span=12, adjust=False).mean()
stock_data['EMA26'] = stock_data['Close'].ewm(span=26, adjust=False).mean()
stock_data['MACD'] = stock_data['EMA12'] - stock_data['EMA26']
stock_data['Signal_Line'] = stock_data['MACD'].ewm(span=9, adjust=False).mean()
 
# 3. Calculate RSI (Relative Strength Index)
delta = stock_data['Close'].diff()
gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
 
rs = gain / loss
stock_data['RSI'] = 100 - (100 / (1 + rs))
 
# 4. Generate trading signals based on MACD and RSI
# Buy when MACD crosses above the signal line and RSI is below 30 (indicating oversold)
stock_data['Buy_Signal'] = np.where((stock_data['MACD'] > stock_data['Signal_Line']) & (stock_data['RSI'] < 30), 1, 0)
 
# Sell when MACD crosses below the signal line and RSI is above 70 (indicating overbought)
stock_data['Sell_Signal'] = np.where((stock_data['MACD'] < stock_data['Signal_Line']) & (stock_data['RSI'] > 70), 1, 0)
 
# 5. Plot stock price and trading signals
plt.figure(figsize=(14, 7))
 
# Plot closing price
plt.subplot(2, 1, 1)
plt.plot(stock_data['Close'], label='Stock Price', color='blue')
plt.scatter(stock_data.index[stock_data['Buy_Signal'] == 1], stock_data['Close'][stock_data['Buy_Signal'] == 1], marker='^', color='green', label='Buy Signal')
plt.scatter(stock_data.index[stock_data['Sell_Signal'] == 1], stock_data['Close'][stock_data['Sell_Signal'] == 1], marker='v', color='red', label='Sell Signal')
plt.title('Stock Price with Buy and Sell Signals')
plt.xlabel('Date')
plt.ylabel('Stock Price (USD)')
plt.legend()
 
# Plot RSI and MACD
plt.subplot(2, 1, 2)
plt.plot(stock_data['RSI'], label='RSI', color='orange')
plt.axhline(70, color='red', linestyle='--', label='Overbought (70)')
plt.axhline(30, color='green', linestyle='--', label='Oversold (30)')
plt.title('RSI (Relative Strength Index)')
plt.xlabel('Date')
plt.ylabel('RSI')
plt.legend()
 
plt.tight_layout()
plt.show()