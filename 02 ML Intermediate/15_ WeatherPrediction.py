"""
Project 55. Weather prediction model
Description:
A weather prediction model forecasts conditions such as temperature based on historical weather data. In this project, we build a regression model to predict the next day's temperature using recent temperature trends. We use a Linear Regression model trained on synthetic weather data.

About:
🌤️ What This Project Demonstrates:
Creates lag features to use recent temperature values for forecasting

Uses Linear Regression for time series prediction

Evaluates model with MSE and R² score

Visualizes predictions over time
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
 
# Generate synthetic temperature data
np.random.seed(42)
days = pd.date_range(start='2023-01-01', periods=100, freq='D')
temperature = 20 + np.sin(np.linspace(0, 3 * np.pi, 100)) * 10 + np.random.normal(0, 2, 100)
 
# Create DataFrame
df = pd.DataFrame({'Date': days, 'Temperature': temperature})
df.set_index('Date', inplace=True)
 
# Feature engineering: past 3 days temperature as input
for i in range(1, 4):
    df[f'Temp_lag_{i}'] = df['Temperature'].shift(i)
 
# Drop NaNs
df.dropna(inplace=True)
 
# Features and target
X = df[['Temp_lag_1', 'Temp_lag_2', 'Temp_lag_3']]
y = df['Temperature']
 
# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)
 
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
plt.plot(y_test.index, y_test.values, label='Actual Temperature')
plt.plot(y_test.index, y_pred, label='Predicted Temperature', linestyle='--')
plt.title("Weather Prediction Model (Temperature)")
plt.xlabel("Date")
plt.ylabel("Temperature (°C)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()