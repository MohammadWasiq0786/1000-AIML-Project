"""
Project 808. Price Optimization Model

Price optimization identifies the best price point for a product to maximize revenue or profit. It analyzes how changes in price affect demand and revenue. In this simplified version, we simulate price vs. demand data and find the price that maximizes revenue using a regression-based approach.

This model fits a quadratic curve to simulate the classic price-revenue relationship and identifies the price point where revenue is maximized. For more advanced setups, you can incorporate cost, customer segments, and price elasticity models.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
 
# Simulated data: prices and corresponding customer demand
data = {
    'Price': [10, 12, 14, 16, 18, 20, 22, 24],
    'Demand': [220, 200, 185, 160, 140, 120, 100, 85]
}
 
df = pd.DataFrame(data)
 
# Calculate revenue (Price * Demand)
df['Revenue'] = df['Price'] * df['Demand']
 
# Fit a quadratic regression model to find the price that maximizes revenue
# Create polynomial features: Price and Price^2
df['Price_Sq'] = df['Price'] ** 2
X = df[['Price', 'Price_Sq']]
y = df['Revenue']
 
model = LinearRegression()
model.fit(X, y)
 
# Generate prices to evaluate model
price_range = np.linspace(10, 25, 100)
price_sq = price_range ** 2
X_pred = pd.DataFrame({'Price': price_range, 'Price_Sq': price_sq})
predicted_revenue = model.predict(X_pred)
 
# Find the price that gives the highest predicted revenue
optimal_index = np.argmax(predicted_revenue)
optimal_price = price_range[optimal_index]
max_revenue = predicted_revenue[optimal_index]
 
# Plot the revenue curve
plt.figure(figsize=(10, 5))
plt.plot(price_range, predicted_revenue, label='Predicted Revenue')
plt.axvline(optimal_price, color='red', linestyle='--', label=f'Optimal Price: ${optimal_price:.2f}')
plt.title('Price Optimization')
plt.xlabel('Price')
plt.ylabel('Revenue')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
 
# Output result
print(f"Optimal Price: ${optimal_price:.2f}")
print(f"Maximum Predicted Revenue: ${max_revenue:.2f}")