"""
Project 807. Demand Forecasting System

Demand forecasting predicts the quantity of products or services customers will purchase in the future. This is crucial for inventory planning, reducing waste, and meeting customer needs. We’ll use a basic regression model on historical demand data to forecast future demand.

This basic model captures the upward trend in demand and projects it into future weeks. For better accuracy, especially in real businesses, you might include variables like price, seasonality, promotions, and competitor actions.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
 
# Sample weekly demand data
data = {
    'Week': np.arange(1, 13),  # Week numbers
    'Units_Demanded': [120, 130, 125, 140, 135, 150, 160, 170, 165, 180, 175, 190]
}
 
df = pd.DataFrame(data)
 
# Features and target
X = df[['Week']]
y = df['Units_Demanded']
 
# Train a simple linear regression model
model = LinearRegression()
model.fit(X, y)
 
# Forecast demand for the next 4 weeks
future_weeks = np.arange(13, 17).reshape(-1, 1)
future_demand = model.predict(future_weeks)
 
# Combine actual and forecasted data
forecast_df = pd.concat([
    df,
    pd.DataFrame({'Week': future_weeks.flatten(), 'Units_Demanded': future_demand})
])
 
# Plot actual vs forecasted demand
plt.figure(figsize=(10, 5))
plt.plot(forecast_df['Week'], forecast_df['Units_Demanded'], marker='o', label='Demand')
plt.axvline(x=12.5, color='gray', linestyle='--', label='Forecast Start')
plt.title('Weekly Demand Forecast')
plt.xlabel('Week')
plt.ylabel('Units Demanded')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
 
# Display forecasted values
print("Forecasted Demand (Next 4 Weeks):")
print(pd.DataFrame({
    'Week': future_weeks.flatten(),
    'Forecasted Units': future_demand.round(1)
}))