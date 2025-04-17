"""
Project 503: Pairs Trading Strategy
Description:
Pairs trading is a market-neutral strategy that involves identifying two stocks that historically move together (i.e., they are cointegrated) and taking advantage of temporary divergences in their prices. The strategy involves going long on one stock and shorting the other when their price ratio deviates from the historical norm. In this project, we will implement a basic pairs trading strategy using cointegration to select pairs of stocks and generate trading signals.

For real-world applications:

Use cointegration tests to identify pairs of stocks with historically strong relationships.

This strategy can be expanded by adding stop-loss, take-profit, or risk management.

✅ What It Does:
Performs a cointegration test to check if two stocks (Apple and Microsoft) are cointegrated, which means their price relationship is stable over time.

Calculates the spread between the two stocks using linear regression to determine the relationship.

Generates trading signals:

Buy Signal: When the spread is more than one standard deviation below the mean (indicating the spread is too wide).

Sell Signal: When the spread is more than one standard deviation above the mean (indicating the spread is too narrow).

Plots the spread along with the trading signals, showing where the strategy would have generated buy and sell signals.

Key Extensions and Customizations:
Real-time data: Replace historical data with live market data to implement the strategy in real-time using APIs such as Alpaca, Binance, or Yahoo Finance.

Multiple pairs: Extend the strategy to trade multiple pairs of stocks and optimize portfolio allocations.

Risk management: Add stop-loss or take-profit rules to manage risk more effectively.

Backtesting: Implement backtesting to evaluate the performance of the strategy over a historical dataset.
"""

import pandas as pd
import numpy as np
import yfinance as yf
import statsmodels.api as sm
import matplotlib.pyplot as plt
 
# 1. Download historical stock data for two stocks (e.g., Apple and Microsoft)
stock1 = yf.download("AAPL", start="2015-01-01", end="2021-01-01")['Close']
stock2 = yf.download("MSFT", start="2015-01-01", end="2021-01-01")['Close']
 
# 2. Perform cointegration test to check if the stocks are cointegrated
def cointegration_test(series1, series2):
    result = sm.tsa.stattools.coint(series1, series2)
    return result[1]  # p-value
 
# Perform the cointegration test between AAPL and MSFT
p_value = cointegration_test(stock1, stock2)
 
print(f"P-value of cointegration test between AAPL and MSFT: {p_value:.4f}")
 
# If p-value < 0.05, the series are cointegrated and we can proceed
if p_value < 0.05:
    print("The stocks are cointegrated and can be used for pairs trading.")
else:
    print("The stocks are not cointegrated. Consider finding a different pair.")
    
# 3. Calculate the spread between the two stocks (spread = stock1 - beta * stock2)
# Calculate beta using linear regression between the two stocks
X = sm.add_constant(stock2)
model = sm.OLS(stock1, X).fit()
beta = model.params[1]  # Coefficient for stock2
spread = stock1 - beta * stock2
 
# 4. Generate trading signals
# Buy when the spread is below the mean - 1 standard deviation
# Sell when the spread is above the mean + 1 standard deviation
mean_spread = spread.mean()
std_spread = spread.std()
 
# Define entry and exit points
buy_signal = spread < mean_spread - std_spread
sell_signal = spread > mean_spread + std_spread
 
# 5. Plot the spread and the trading signals
plt.figure(figsize=(14, 7))
plt.plot(spread, label='Spread (AAPL - Beta * MSFT)', color='blue')
plt.axhline(mean_spread, color='green', linestyle='--', label='Mean')
plt.axhline(mean_spread - std_spread, color='red', linestyle='--', label='Buy Signal Threshold')
plt.axhline(mean_spread + std_spread, color='purple', linestyle='--', label='Sell Signal Threshold')
plt.scatter(spread.index[buy_signal], spread[buy_signal], marker='^', color='g', label="Buy Signal")
plt.scatter(spread.index[sell_signal], spread[sell_signal], marker='v', color='r', label="Sell Signal")
plt.title("Pairs Trading Strategy (AAPL vs MSFT)")
plt.xlabel('Date')
plt.ylabel('Spread')
plt.legend(loc='best')
plt.show()