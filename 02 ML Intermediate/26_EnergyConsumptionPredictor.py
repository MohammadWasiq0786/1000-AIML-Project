"""
Project 66. Energy consumption predictor
Description:
An energy consumption predictor forecasts future power usage based on past consumption data and external factors like time or weather. This helps optimize grid operations, billing, and energy-saving strategies. In this project, we build a regression model to predict energy consumption using lag features.

About:
⚡ What This Project Demonstrates:
Predicts future energy usage using lag-based time series features

Uses Linear Regression for continuous value forecasting

Visualizes actual vs predicted energy for insights
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
 
# Simulated hourly energy consumption data
np.random.seed(42)
hours = pd.date_range(start='2023-01-01', periods=200, freq='H')
energy = 50 + 10 * np.sin(np.linspace(0, 4 * np.pi, 200)) + np.random.normal(0, 3, 200)
 
# Create DataFrame
df = pd.DataFrame({'Timestamp': hours, 'EnergyConsumption': energy})
df.set_index('Timestamp', inplace=True)
 
# Create lag features
for lag in range(1, 4):
    df[f'Lag_{lag}'] = df['EnergyConsumption'].shift(lag)
 
# Drop rows with NaNs due to lagging
df.dropna(inplace=True)
 
# Define features and target
X = df[['Lag_1', 'Lag_2', 'Lag_3']]
y = df['EnergyConsumption']
 
# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.25)
 
# Train model
model = LinearRegression()
model.fit(X_train, y_train)
 
# Predict and evaluate
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
 
print(f"Mean Squared Error: {mse:.2f}")
print(f"R² Score: {r2:.2f}")
 
# Plot predictions
plt.figure(figsize=(10, 5))
plt.plot(y_test.index, y_test.values, label='Actual Energy Consumption')
plt.plot(y_test.index, y_pred, linestyle='--', label='Predicted Energy Consumption')
plt.title("Energy Consumption Forecast")
plt.xlabel("Time")
plt.ylabel("kWh")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()