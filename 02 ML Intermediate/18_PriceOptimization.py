"""
Project 58. Price optimization tool
Description:
A price optimization tool helps businesses determine the ideal price point for products to maximize revenue or profit. In this project, we simulate a dataset of product prices and sales, and build a regression model to understand the price-demand relationship and identify the optimal price for maximum revenue.

About:
💡 What This Project Demonstrates:
Simulates price vs demand and calculates revenue

Builds a linear model to estimate the price-demand curve

Identifies the optimal price that maximizes predicted revenue

Visualizes the revenue curve and optimal price point
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
 
# Simulated price vs demand data
np.random.seed(42)
prices = np.linspace(5, 50, 20)
demand = 1000 - 15 * prices + np.random.normal(0, 50, len(prices))  # demand drops as price increases
revenue = prices * demand
 
# Create DataFrame
df = pd.DataFrame({'Price': prices, 'Demand': demand, 'Revenue': revenue})
 
# Train regression model to predict demand from price
X = df[['Price']]
y = df['Demand']
model = LinearRegression()
model.fit(X, y)
 
# Predict demand for a range of prices
price_range = np.linspace(5, 50, 100).reshape(-1, 1)
predicted_demand = model.predict(price_range)
predicted_revenue = price_range.flatten() * predicted_demand
 
# Find optimal price (price with max predicted revenue)
optimal_index = np.argmax(predicted_revenue)
optimal_price = price_range.flatten()[optimal_index]
max_revenue = predicted_revenue[optimal_index]
 
# Display optimal result
print(f"Optimal Price: ${optimal_price:.2f}")
print(f"Expected Revenue at Optimal Price: ${max_revenue:.2f}")
 
# Plot revenue vs price
plt.figure(figsize=(8, 5))
plt.plot(price_range, predicted_revenue, label='Predicted Revenue')
plt.axvline(optimal_price, color='red', linestyle='--', label=f'Optimal Price: ${optimal_price:.2f}')
plt.title("Price Optimization: Revenue vs Price")
plt.xlabel("Price ($)")
plt.ylabel("Predicted Revenue ($)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()