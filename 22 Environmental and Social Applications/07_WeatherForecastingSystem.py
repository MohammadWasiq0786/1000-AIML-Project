"""
Project 847: Weather Forecasting System
Description
Weather forecasting uses historical meteorological data to predict future conditions like temperature, humidity, rainfall, and wind. In this project, we simulate daily weather data and build a multi-output regression model to forecast next day’s weather (temperature, humidity, wind speed, and rainfall).

✅ Uses of this model:

Smart farming tools

Local weather station dashboards

Energy optimization (solar, wind, HVAC)
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
 
# Simulate daily weather conditions
np.random.seed(42)
n_samples = 1000
 
# Current day weather features
temp_today = np.random.normal(25, 5, n_samples)
humidity_today = np.random.normal(60, 10, n_samples)
wind_today = np.random.normal(10, 2, n_samples)
rain_today = np.random.normal(2, 1.5, n_samples)
 
# Simulate next-day values with a bit of trend and noise
temp_next = temp_today + np.random.normal(0, 1.5, n_samples)
humidity_next = humidity_today + np.random.normal(0, 5, n_samples)
wind_next = wind_today + np.random.normal(0, 1, n_samples)
rain_next = rain_today + np.random.normal(0, 1, n_samples)
 
# Feature matrix and multi-output labels
X = np.stack([temp_today, humidity_today, wind_today, rain_today], axis=1)
y = np.stack([temp_next, humidity_next, wind_next, rain_next], axis=1)
 
# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
 
# Build multi-output regression model
model = models.Sequential([
    layers.Input(shape=(4,)),
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(4)  # Output: temperature, humidity, wind, rainfall
])
 
model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=0)
 
# Evaluate performance
loss, mae = model.evaluate(X_test, y_test)
print(f"✅ Weather Forecasting MAE (Temp, Humidity, Wind, Rain): {mae}")
 
# Predict and display 5 samples
preds = model.predict(X_test[:5])
for i in range(5):
    print(f"Sample {i+1}:")
    print(f"  🌡️ Temp: {preds[i][0]:.1f}°C | 💧 Humidity: {preds[i][1]:.1f}% | 🌬️ Wind: {preds[i][2]:.1f} km/h | 🌧️ Rain: {preds[i][3]:.1f} mm")