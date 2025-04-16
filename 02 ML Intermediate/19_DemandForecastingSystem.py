"""
Project 59. Demand forecasting system
Description:
A demand forecasting system predicts future product demand based on historical sales data. This is crucial for inventory management, logistics, and production planning. In this project, we build a simple time series forecasting model using Linear Regression with lag features to forecast future demand.

About:
📦 What This Project Demonstrates:
Uses lag features for time series regression forecasting

Predicts future product demand using historical sales data

Visualizes forecasted vs actual demand

Evaluates model with MSE and R² score
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
 
# Generate synthetic monthly demand data
np.random.seed(42)
months = pd.date_range(start='2022-01-01', periods=24, freq='M')
demand = 500 + np.sin(np.linspace(0, 3 * np.pi, 24)) * 100 + np.random.normal(0, 20, 24)
 
# Create DataFrame
df = pd.DataFrame({'Month': months, 'Demand': demand})
df.set_index('Month', inplace=True)
 
# Create lag features (previous 3 months)
for i in range(1, 4):
    df[f'Lag_{i}'] = df['Demand'].shift(i)
 
# Drop NaNs due to lagging
df.dropna(inplace=True)
 
# Features and target
X = df[['Lag_1', 'Lag_2', 'Lag_3']]
y = df['Demand']
 
# Split train and test
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
 
# Plot actual vs predicted demand
plt.figure(figsize=(10, 5))
plt.plot(y_test.index, y_test.values, label='Actual Demand')
plt.plot(y_test.index, y_pred, label='Predicted Demand', linestyle='--')
plt.title("Demand Forecasting Model")
plt.xlabel("Month")
plt.ylabel("Units Sold")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()