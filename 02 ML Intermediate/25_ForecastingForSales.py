"""
Project 65. Time series forecasting for sales
Description:
Sales forecasting predicts future sales based on historical trends, helping businesses make informed decisions on inventory, staffing, and promotions. In this project, we simulate monthly sales data and use Linear Regression with lag features to forecast future sales.

About:
🛒 What This Project Demonstrates:
Forecasts future sales using lag-based features

Applies Linear Regression to time series data

Visualizes and evaluates results using MSE and R² Score
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
 
# Simulated monthly sales data
np.random.seed(42)
months = pd.date_range(start='2022-01-01', periods=24, freq='M')
sales = 1000 + np.sin(np.linspace(0, 3 * np.pi, 24)) * 200 + np.random.normal(0, 50, 24)
 
# Create DataFrame
df = pd.DataFrame({'Month': months, 'Sales': sales})
df.set_index('Month', inplace=True)
 
# Generate lag features (last 3 months of sales)
for lag in range(1, 4):
    df[f'Lag_{lag}'] = df['Sales'].shift(lag)
 
# Drop missing values
df.dropna(inplace=True)
 
# Features and target
X = df[['Lag_1', 'Lag_2', 'Lag_3']]
y = df['Sales']
 
# Train-test split
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
 
# Plot actual vs predicted
plt.figure(figsize=(10, 5))
plt.plot(y_test.index, y_test.values, label='Actual Sales')
plt.plot(y_test.index, y_pred, label='Predicted Sales', linestyle='--')
plt.title("Sales Forecasting using Time Series Regression")
plt.xlabel("Month")
plt.ylabel("Sales")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()