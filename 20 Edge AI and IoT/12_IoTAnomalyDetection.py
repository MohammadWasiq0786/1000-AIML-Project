"""
Project 772: IoT Anomaly Detection
Description
IoT anomaly detection identifies unusual patterns in sensor data that could indicate faults, intrusions, or failures. It’s vital for predictive maintenance, security, and real-time monitoring in industrial and smart home systems. We’ll simulate this using synthetic sensor data and build an autoencoder to detect anomalies based on reconstruction error.

This project demonstrates real-time anomaly detection using edge-compatible models. With proper data pipeline integration, this approach can run on low-power devices in smart factories, homes, or critical infrastructure.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
 
# Simulate IoT sensor data: mostly normal data with a few anomalies
np.random.seed(42)
 
# Generate normal data (e.g., temperature readings between 20-25°C)
normal_data = np.random.normal(loc=22.5, scale=1.0, size=(1000, 1))
 
# Add a few anomalies (e.g., spikes)
anomalies = np.random.normal(loc=30, scale=1.0, size=(50, 1))
data = np.concatenate([normal_data, anomalies], axis=0)
 
# Shuffle the dataset
np.random.shuffle(data)
 
# Normalize the data
data_min, data_max = data.min(), data.max()
data_scaled = (data - data_min) / (data_max - data_min)
 
# Build a simple autoencoder for 1D data
def build_autoencoder():
    model = models.Sequential([
        layers.Input(shape=(1,)),
        layers.Dense(4, activation='relu'),
        layers.Dense(1)
    ])
    return model
 
# Create and train the autoencoder
autoencoder = build_autoencoder()
autoencoder.compile(optimizer='adam', loss='mse')
autoencoder.fit(data_scaled, data_scaled, epochs=20, batch_size=32, verbose=0)
 
# Predict (reconstruct) and calculate reconstruction error
reconstructed = autoencoder.predict(data_scaled)
errors = np.abs(data_scaled - reconstructed)
 
# Define anomaly threshold (e.g., mean + 3*std of errors on normal data)
threshold = np.mean(errors) + 3 * np.std(errors)
 
# Flag anomalies
anomaly_flags = errors > threshold
 
# Print detection results
num_anomalies_detected = np.sum(anomaly_flags)
print(f"✅ Detected {num_anomalies_detected} anomalies using autoencoder-based thresholding.")
 
# Optional: Plot reconstruction error
plt.figure(figsize=(10, 4))
plt.plot(errors, label='Reconstruction Error')
plt.axhline(y=threshold, color='red', linestyle='--', label='Anomaly Threshold')
plt.title('IoT Anomaly Detection: Reconstruction Error')
plt.xlabel('Sample Index')
plt.ylabel('Error')
plt.legend()
plt.show()