"""
Project 775: Environmental Monitoring System
Description
An environmental monitoring system collects data such as temperature, humidity, air quality, and noise levels to track environmental conditions in smart homes, greenhouses, or cities. We simulate such a system with synthetic sensor readings and use a regression model to predict air quality index (AQI) based on multiple sensor inputs.

This model can power real-world environmental dashboards or trigger alerts on edge devices when AQI crosses unsafe thresholds. With real sensor data (e.g., from BME280 or CCS811), it can run on Raspberry Pi or ESP32 boards.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
 
# Simulate sensor data: temp (°C), humidity (%), noise (dB), CO2 (ppm)
np.random.seed(42)
n_samples = 1000
 
temperature = np.random.normal(22, 2, n_samples)
humidity = np.random.normal(50, 10, n_samples)
noise = np.random.normal(40, 5, n_samples)
co2 = np.random.normal(600, 100, n_samples)
 
# Simulate AQI (as a function of the sensors with added noise)
aqi = (0.5 * temperature + 0.3 * humidity + 0.2 * noise + 0.4 * (co2 / 100)) + np.random.normal(0, 5, n_samples)
 
# Stack sensor features
X = np.stack([temperature, humidity, noise, co2], axis=1)
y = aqi
 
# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
 
# Build a simple regression model
model = models.Sequential([
    layers.Input(shape=(4,)),
    layers.Dense(32, activation='relu'),
    layers.Dense(16, activation='relu'),
    layers.Dense(1)  # AQI output
])
 
model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=0)
 
# Evaluate the model
loss, mae = model.evaluate(X_test, y_test)
print(f"✅ Environmental Monitoring Model MAE: {mae:.2f} (AQI units)")
 
# Predict and plot results
predicted_aqi = model.predict(X_test).flatten()
plt.figure(figsize=(10, 4))
plt.plot(y_test[:100], label='Actual AQI')
plt.plot(predicted_aqi[:100], label='Predicted AQI')
plt.title("Environmental Monitoring: AQI Prediction")
plt.xlabel("Sample Index")
plt.ylabel("AQI")
plt.legend()
plt.show()