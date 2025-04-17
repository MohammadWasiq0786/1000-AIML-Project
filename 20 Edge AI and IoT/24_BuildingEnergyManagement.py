"""
Project 784: Building Energy Management
Description
A Building Energy Management System (BEMS) monitors and controls energy use within buildings to increase efficiency, reduce costs, and lower environmental impact. This project simulates sensor readings and usage data from HVAC, lighting, and appliances, and uses a regression model to predict total daily energy consumption for a building.

This model can be deployed on building management dashboards or integrated into BMS software to enable automated control systems that adjust HVAC/lighting schedules based on forecasted usage.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
 
# Simulate building sensor readings: HVAC runtime (hrs), lighting usage (hrs), occupancy (people), outside temp (°C)
np.random.seed(42)
n_samples = 1000
 
hvac_runtime = np.random.normal(5, 2, n_samples)           # hours per day
lighting_usage = np.random.normal(6, 1.5, n_samples)       # hours per day
occupancy = np.random.randint(1, 20, n_samples)            # people present
outside_temp = np.random.normal(25, 5, n_samples)          # Celsius
 
# Simulate total energy usage (kWh) with some weighted influence
total_energy = (hvac_runtime * 3.5 + lighting_usage * 1.2 + occupancy * 0.6 +
                (35 - outside_temp) * 0.8 + np.random.normal(0, 2, n_samples))
 
# Feature matrix and labels
X = np.stack([hvac_runtime, lighting_usage, occupancy, outside_temp], axis=1)
y = total_energy
 
# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
 
# Build a regression model
model = models.Sequential([
    layers.Input(shape=(4,)),
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(1)  # Predicted energy usage
])
 
model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=0)
 
# Evaluate
loss, mae = model.evaluate(X_test, y_test)
print(f"✅ Building Energy Model MAE: {mae:.2f} kWh")
 
# Predict and visualize
preds = model.predict(X_test[:50]).flatten()
actual = y_test[:50]
 
plt.figure(figsize=(10, 4))
plt.plot(actual, label='Actual')
plt.plot(preds, label='Predicted')
plt.title("Building Energy Usage Prediction")
plt.xlabel("Sample Index")
plt.ylabel("kWh")
plt.legend()
plt.show()