"""
Project 844: Renewable Energy Forecasting
Description
Renewable energy forecasting is crucial for integrating solar, wind, and hydro power into the grid. It helps balance supply-demand, reduce energy waste, and stabilize grids. In this project, we simulate environmental factors and build a regression model to forecast renewable energy output (in MW), assuming a mixed source system (solar + wind).

✅ This type of model powers:

Smart grid controllers

Renewable farm operation tools

Demand-response systems for sustainability and cost reduction
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
 
# Simulate environmental features: solar irradiance (W/m²), wind speed (m/s), temperature (°C), humidity (%)
np.random.seed(42)
n_samples = 1000
 
solar_irradiance = np.random.normal(600, 100, n_samples)
wind_speed = np.random.normal(7, 2, n_samples)
temperature = np.random.normal(25, 5, n_samples)
humidity = np.random.normal(50, 10, n_samples)
 
# Simulate renewable energy output (MW)
# solar output mainly depends on irradiance and temperature; wind depends on speed
energy_output = (
    0.015 * solar_irradiance +
    0.2 * wind_speed ** 3 -  # wind power ~ cube of speed
    0.05 * (temperature - 25) +
    np.random.normal(0, 2, n_samples)  # noise
)
 
# Feature matrix and labels
X = np.stack([solar_irradiance, wind_speed, temperature, humidity], axis=1)
y = energy_output
 
# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
 
# Build regression model
model = models.Sequential([
    layers.Input(shape=(4,)),
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(1)  # output: energy forecast (MW)
])
 
model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=0)
 
# Evaluate performance
loss, mae = model.evaluate(X_test, y_test)
print(f"✅ Renewable Energy Forecast MAE: {mae:.2f} MW")
 
# Predict and plot
predictions = model.predict(X_test[:50]).flatten()
plt.figure(figsize=(10, 4))
plt.plot(y_test[:50], label="Actual")
plt.plot(predictions, label="Predicted")
plt.title("Renewable Energy Output Forecast")
plt.xlabel("Sample Index")
plt.ylabel("Energy Output (MW)")
plt.legend()
plt.grid(True)
plt.show()