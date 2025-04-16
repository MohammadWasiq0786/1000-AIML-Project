"""
Project 56. Traffic prediction system
Description:
A traffic prediction system forecasts future vehicle flow or congestion levels based on historical traffic data. In this project, we build a simple regression-based model that predicts hourly vehicle counts using previous observations. This can support smart transportation planning and congestion management.

About:
🚗 What This Project Demonstrates:
Forecasts vehicle flow using recent history (lag features)

Uses Linear Regression for hourly time series prediction

Evaluates predictions with MSE and R²

Visualizes predicted vs actual traffic data
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
 
# Generate synthetic hourly traffic data
np.random.seed(42)
hours = pd.date_range(start='2023-01-01 00:00:00', periods=200, freq='H')
traffic_volume = 200 + 100 * np.sin(np.linspace(0, 4 * np.pi, 200)) + np.random.normal(0, 20, 200)
 
# Create DataFrame
df = pd.DataFrame({'Timestamp': hours, 'TrafficVolume': traffic_volume})
df.set_index('Timestamp', inplace=True)
 
# Create lag features (previous 3 hours)
for i in range(1, 4):
    df[f'Lag_{i}'] = df['TrafficVolume'].shift(i)
 
# Drop NaN values
df.dropna(inplace=True)
 
# Define features and target
X = df[['Lag_1', 'Lag_2', 'Lag_3']]
y = df['TrafficVolume']
 
# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)
 
# Train the model
model = LinearRegression()
model.fit(X_train, y_train)
 
# Predict and evaluate
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
 
print(f"Mean Squared Error: {mse:.2f}")
print(f"R² Score: {r2:.2f}")
 
# Plot actual vs predicted traffic volume
plt.figure(figsize=(10, 5))
plt.plot(y_test.index, y_test.values, label='Actual Traffic')
plt.plot(y_test.index, y_pred, linestyle='--', label='Predicted Traffic')
plt.title("Traffic Volume Prediction")
plt.xlabel("Time")
plt.ylabel("Vehicle Count")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()