"""
Project 501: High-Frequency Trading Simulation
Description:
High-frequency trading (HFT) involves executing a large number of trades at extremely fast speeds, typically leveraging algorithmic strategies to exploit small price movements. In this project, we will simulate an HFT strategy using moving averages and order book data to perform real-time trading simulations based on predefined rules.

For real-world systems:

Integrate with live market data (e.g., from Alpaca, Interactive Brokers, or Binance API).

Implement more advanced strategies like statistical arbitrage or market-making.

✅ What It Does:
Simulates high-frequency price data (price data changing every 5 minutes).

Implements a simple moving average crossover strategy:

Buy when the short-term moving average crosses above the long-term moving average.

Sell when the short-term moving average crosses below the long-term moving average.

Tracks the balance and number of shares held as the strategy executes buy/sell signals.

Visualizes the trading strategy performance by plotting the stock price along with the moving averages and buy/sell signals.

Key Extensions and Customizations:
Real-time data: Replace simulated data with live market data from Alpaca, Binance, or Interactive Brokers using their APIs.

Advanced strategies: Implement more advanced strategies like statistical arbitrage, momentum trading, or market-making.

Backtesting: Extend the project with a backtesting framework to evaluate performance over multiple years of data.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
 
# 1. Simulate high-frequency stock price data
np.random.seed(42)
 
# Simulating 5-minute intervals for 1 day (288 intervals)
time_steps = 288
price_data = 100 + np.cumsum(np.random.randn(time_steps))  # Random walk for stock prices
 
# Create a DataFrame with simulated price data
timestamps = pd.date_range("2023-01-01", periods=time_steps, freq="5T")
df = pd.DataFrame({"timestamp": timestamps, "price": price_data})
 
# 2. Define the trading strategy: Simple moving average crossover
short_window = 10  # Short window for fast moving average
long_window = 50   # Long window for slow moving average
 
df['short_mavg'] = df['price'].rolling(window=short_window, min_periods=1).mean()
df['long_mavg'] = df['price'].rolling(window=long_window, min_periods=1).mean()
 
# 3. Define buy/sell signals based on crossover strategy
df['signal'] = 0  # Default: no action
df['signal'][short_window:] = np.where(df['short_mavg'][short_window:] > df['long_mavg'][short_window:], 1, 0)  # Buy signal
df['position'] = df['signal'].diff()
 
# 4. Simulate the trading actions (buy = 1, sell = -1)
initial_balance = 10000  # Starting with $10,000
balance = initial_balance
shares_held = 0
buy_sell_history = []
 
for i, row in df.iterrows():
    if row['position'] == 1:  # Buy signal
        shares_held = balance // row['price']  # Buy as many shares as possible
        balance -= shares_held * row['price']  # Deduct the balance
        buy_sell_history.append(('Buy', row['timestamp'], row['price'], shares_held))
    elif row['position'] == -1:  # Sell signal
        balance += shares_held * row['price']  # Sell all shares
        buy_sell_history.append(('Sell', row['timestamp'], row['price'], shares_held))
        shares_held = 0  # Reset shares held after selling
 
# 5. Final balance
final_balance = balance + (shares_held * df['price'].iloc[-1])  # Add remaining shares value
 
# 6. Plot the results
plt.figure(figsize=(12, 6))
plt.plot(df['timestamp'], df['price'], label="Stock Price")
plt.plot(df['timestamp'], df['short_mavg'], label="Short Moving Average (10)")
plt.plot(df['timestamp'], df['long_mavg'], label="Long Moving Average (50)")
plt.scatter(df['timestamp'][df['position'] == 1], df['price'][df['position'] == 1], marker='^', color='g', label="Buy Signal")
plt.scatter(df['timestamp'][df['position'] == -1], df['price'][df['position'] == -1], marker='v', color='r', label="Sell Signal")
plt.title("High-Frequency Trading Strategy (Moving Average Crossover)")
plt.xlabel("Timestamp")
plt.ylabel("Price")
plt.legend(loc="best")
plt.grid(True)
plt.show()
 
print(f"Initial Balance: ${initial_balance}")
print(f"Final Balance: ${final_balance:.2f}")
print(f"Total Profit: ${final_balance - initial_balance:.2f}")