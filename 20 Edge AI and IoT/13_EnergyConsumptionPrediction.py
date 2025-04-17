"""
Project 773: Energy Consumption Prediction
Description
Energy consumption prediction helps optimize power usage and reduce costs in smart homes, buildings, and grids. By forecasting future consumption based on historical data, IoT systems can plan operations more efficiently. We'll build a simple LSTM model to predict future energy usage from synthetic time series data.

This model can be deployed on edge devices (like a Raspberry Pi or ESP32 with TensorFlow Lite) to recognize user activity in real-time based on sensor fusion inputs.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
 
# Simulate energy consumption data (e.g., daily kWh readings)
np.random.seed(42)
days = 500
energy_data = np.sin(np.arange(days) * 2 * np.pi / 30) + np.random.normal(0, 0.2, size=days)
energy_data = energy_data.reshape(-1, 1)
 
# Normalize the data
min_val, max_val = energy_data.min(), energy_data.max()
normalized_data = (energy_data - min_val) / (max_val - min_val)
 
# Create sequences for LSTM input
def create_sequences(data, window_size=30):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i+window_size])
        y.append(data[i+window_size])
    return np.array(X), np.array(y)
 
# Prepare sequences
window_size = 30
X, y = create_sequences(normalized_data, window_size)
 
# Split into train and test sets
split = int(0.8 * len(X))
X_train, y_train = X[:split], y[:split]
X_test, y_test = X[split:], y[split:]
 
# Build LSTM model
model = models.Sequential([
    layers.LSTM(32, input_shape=(window_size, 1)),
    layers.Dense(1)
])
 
model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, epochs=10, batch_size=16, verbose=0)
 
# Predict on test data
predictions = model.predict(X_test)
 
# Reverse normalization for plotting
def denormalize(x): return x * (max_val - min_val) + min_val
predicted_energy = denormalize(predictions)
actual_energy = denormalize(y_test)
 
# Plot the prediction
plt.figure(figsize=(10, 4))
plt.plot(actual_energy, label="Actual")
plt.plot(predicted_energy, label="Predicted")
plt.title("Energy Consumption Prediction")
plt.xlabel("Time Step")
plt.ylabel("Energy (kWh)")
plt.legend()
plt.show()
 
print("✅ LSTM model trained and tested for energy prediction.")