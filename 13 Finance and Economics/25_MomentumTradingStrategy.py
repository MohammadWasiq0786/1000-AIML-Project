"""
Project 505: Momentum Trading Strategy
Description:
The momentum trading strategy is based on the idea that assets that have performed well in the past will continue to perform well in the future, and those that have performed poorly will continue to underperform. This strategy involves buying assets with strong upward momentum and selling assets with downward momentum. In this project, we will implement a simple momentum strategy using the rate of change (ROC) to generate trading signals.

For real-world applications:

Use technical indicators like Relative Strength Index (RSI), Moving Average Convergence Divergence (MACD), or average directional index (ADX) to refine the strategy.

Backtest the strategy on multiple assets to evaluate its effectiveness.

About:
✅ What It Does:
Calculates the Rate of Change (ROC) for the stock price over a specified period (14 days in this case). The ROC measures the percentage change in price over a specified time frame.

Generates trading signals:

Buy signal when the ROC is positive, indicating upward momentum.

Sell signal when the ROC is negative, indicating downward momentum.

Plots the stock price and the ROC, showing the buy and sell signals.

Key Extensions and Customizations:
Adjust window size: Experiment with different window sizes (e.g., 30-day, 60-day) to adapt to different time frames and trading strategies.

Real-time data: Replace historical data with live market data from APIs like Alpaca, Binance, or Yahoo Finance to implement real-time momentum trading.

Enhance with additional indicators: Combine MACD, RSI, or Bollinger Bands with momentum strategies to improve accuracy and reduce false signals.

Backtesting: Implement backtesting frameworks to assess the performance of the momentum strategy over multiple assets and historical data.

Risk management: Add stop-loss, take-profit, and position sizing for better risk control.
"""

import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
 
# 1. Download historical stock data (e.g., Apple stock)
stock_data = yf.download("AAPL", start="2018-01-01", end="2021-01-01")['Close']
 
# 2. Calculate the Rate of Change (ROC)
window = 14  # 14-day window
roc = stock_data.pct_change(periods=window) * 100  # Rate of Change in percentage
 
# 3. Generate buy and sell signals based on ROC
buy_signal = roc > 0  # Buy when ROC is positive (indicating upward momentum)
sell_signal = roc < 0  # Sell when ROC is negative (indicating downward momentum)
 
# 4. Plot the stock price and momentum signals
plt.figure(figsize=(14, 7))
 
# Plot the stock price
plt.subplot(2, 1, 1)
plt.plot(stock_data, label="Stock Price", color='blue')
plt.scatter(stock_data.index[buy_signal], stock_data[buy_signal], marker='^', color='green', label="Buy Signal")
plt.scatter(stock_data.index[sell_signal], stock_data[sell_signal], marker='v', color='red', label="Sell Signal")
plt.title("Momentum Trading Strategy - Stock Price")
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
 
# Plot the Rate of Change (ROC)
plt.subplot(2, 1, 2)
plt.plot(roc, label="Rate of Change (ROC)", color='purple')
plt.axhline(0, color='black', linestyle='--', label="Zero Line")
plt.title("Rate of Change for Momentum Trading Strategy")
plt.xlabel('Date')
plt.ylabel('ROC (%)')
plt.legend()
 
plt.tight_layout()
plt.show()