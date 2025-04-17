"""
Project 504: Mean Reversion Strategy Implementation
Description:
A mean reversion strategy is based on the idea that asset prices tend to revert to their historical average over time. This strategy involves buying an asset when its price falls below a certain threshold (indicating it’s undervalued) and selling it when the price exceeds the threshold (indicating it’s overvalued). In this project, we will implement a simple mean reversion strategy using the Z-score to generate buy and sell signals.

For real-world applications:

Implement this strategy on multiple assets (stocks, forex, or commodities).

Enhance this strategy by adding risk management and position sizing for better performance.

✅ What It Does:
Calculates the Z-score for the stock price using a rolling window (20 days in this case).

Z-score measures how far the current price is from the mean, in terms of standard deviations.

Generates trading signals:

Buy signal when the Z-score is below -1 (indicating the stock price is significantly lower than its historical average).

Sell signal when the Z-score is above 1 (indicating the stock price is significantly higher than its historical average).

Plots the stock price, Z-score, and trading signals to visualize the strategy.

Key Extensions and Customizations:
Adjust the window size: Experiment with different rolling window sizes (e.g., 30-day, 50-day) for more robust signals.

Real-time data: Integrate live market data from APIs like Alpaca, Binance, or Yahoo Finance to implement the strategy in real time.

Multiple assets: Extend this strategy to multiple stocks or other asset classes (e.g., commodities, currencies).

Risk management: Add stop-loss, take-profit, and position sizing for more controlled risk exposure.

Backtesting: Implement backtesting to evaluate the historical performance of the strategy.
"""

import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
 
# 1. Download historical stock data (e.g., Apple stock)
stock_data = yf.download("AAPL", start="2018-01-01", end="2021-01-01")['Close']
 
# 2. Calculate the rolling mean and rolling standard deviation
window = 20  # 20-day window
rolling_mean = stock_data.rolling(window=window).mean()
rolling_std = stock_data.rolling(window=window).std()
 
# 3. Calculate the Z-score
z_score = (stock_data - rolling_mean) / rolling_std
 
# 4. Generate buy and sell signals
buy_signal = z_score < -1  # Buy when the Z-score is below -1 (indicating oversold)
sell_signal = z_score > 1  # Sell when the Z-score is above 1 (indicating overbought)
 
# 5. Plot the stock price, Z-score, and trading signals
plt.figure(figsize=(14, 7))
 
# Plot the stock price
plt.subplot(2, 1, 1)
plt.plot(stock_data, label="Stock Price", color='blue')
plt.plot(rolling_mean, label="Rolling Mean (20-day)", color='orange', linestyle='--')
plt.scatter(stock_data.index[buy_signal], stock_data[buy_signal], marker='^', color='green', label="Buy Signal")
plt.scatter(stock_data.index[sell_signal], stock_data[sell_signal], marker='v', color='red', label="Sell Signal")
plt.title("Mean Reversion Strategy (Z-score) - Stock Price")
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
 
# Plot the Z-score
plt.subplot(2, 1, 2)
plt.plot(z_score, label="Z-score", color='purple')
plt.axhline(1, color='red', linestyle='--', label="Sell Threshold (Z=1)")
plt.axhline(-1, color='green', linestyle='--', label="Buy Threshold (Z=-1)")
plt.title("Z-score for Mean Reversion Strategy")
plt.xlabel('Date')
plt.ylabel('Z-score')
plt.legend()
 
plt.tight_layout()
plt.show()