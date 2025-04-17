"""
Project 507: Financial Stress Test Simulation
Description:
A financial stress test simulates extreme economic conditions (e.g., a financial crisis or market crash) to evaluate the resilience of financial institutions, portfolios, or individual assets. In this project, we will simulate a stress test by applying shocks to key financial variables, such as stock prices, interest rates, and market volatility, to assess the impact on a hypothetical portfolio.

For real-world applications:

Use historical financial crises data (e.g., 2008 Financial Crisis) to simulate stress scenarios.

You can also integrate Monte Carlo simulations or Scenario Analysis to evaluate different stress scenarios.

About:

✅ What It Does:
Simulates a portfolio with two assets: stock and bond, with assumed returns and volatilities.

Simulates a stress test by applying shocks (e.g., a 30% stock price drop and 10% bond price drop due to an interest rate hike).

Visualizes the portfolio value before and after the stress test, showing the impact of the shocks.

Calculates the drop in portfolio value due to the stress scenario.

Key Extensions and Customizations:
Use real historical data: Replace simulated data with actual asset returns from Yahoo Finance or other financial data sources.

Multiple stress scenarios: Implement different stress scenarios such as economic downturns, geopolitical events, or liquidity crises.

Monte Carlo simulation: Use Monte Carlo methods to simulate a range of potential stress test scenarios based on historical volatility.

Portfolio optimization: Extend the model to optimize the portfolio's risk-return trade-off and stress-test different portfolio compositions.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
 
# 1. Simulate a portfolio of assets (e.g., stocks and bonds)
np.random.seed(42)
asset_returns = {
    'Stock': np.random.normal(0.08, 0.15, 1000),  # 8% annual return, 15% volatility
    'Bond': np.random.normal(0.03, 0.05, 1000)    # 3% annual return, 5% volatility
}
 
# Create a DataFrame for portfolio assets
df = pd.DataFrame(asset_returns)
 
# 2. Simulate portfolio value (initial value of $100,000)
initial_portfolio_value = 100000
df['Portfolio Value'] = initial_portfolio_value * (1 + df['Stock'] + df['Bond']).cumprod()
 
# 3. Simulate a financial stress scenario (e.g., stock market crash, interest rate hike)
# Apply shocks: 30% stock price drop, 10% interest rate increase (shock values can vary)
shock_stock = 0.30  # 30% drop in stock price
shock_bond = -0.10  # 10% drop in bond price due to rate hike
 
# Apply shocks to the portfolio value (stress scenario)
df['Portfolio Value Shocked'] = df['Portfolio Value'] * (1 - shock_stock + shock_bond)
 
# 4. Plot portfolio value before and after stress test
plt.figure(figsize=(12, 6))
plt.plot(df['Portfolio Value'], label='Original Portfolio Value', color='blue')
plt.plot(df['Portfolio Value Shocked'], label='Portfolio Value After Stress Test', color='red', linestyle='--')
plt.title("Financial Stress Test Simulation: Portfolio Value Before and After Shock")
plt.xlabel("Time (Simulation Period)")
plt.ylabel("Portfolio Value (USD)")
plt.legend(loc='best')
plt.grid(True)
plt.show()
 
# 5. Calculate the impact of the stress test on portfolio value
portfolio_drop = df['Portfolio Value'].iloc[-1] - df['Portfolio Value Shocked'].iloc[-1]
percent_drop = (portfolio_drop / df['Portfolio Value'].iloc[-1]) * 100
 
print(f"Portfolio Value Drop Due to Stress Test: ${portfolio_drop:.2f}")
print(f"Percentage Drop in Portfolio Value: {percent_drop:.2f}%")