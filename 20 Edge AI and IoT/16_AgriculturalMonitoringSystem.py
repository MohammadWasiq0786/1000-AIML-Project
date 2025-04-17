"""
Project 776: Agricultural Monitoring System
Description
An agricultural monitoring system uses IoT sensors to monitor conditions like soil moisture, temperature, humidity, and sunlight. It helps automate irrigation and crop management. In this project, we simulate sensor data and build a binary classifier to predict “Water Needed” vs. “No Water Needed” — ideal for smart irrigation systems.

This model can power a low-cost, solar-powered smart irrigation system, using real sensor modules like DHT11, soil probes, and LDRs, running on ESP32 or Arduino.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
 
# Simulate sensor readings
np.random.seed(42)
n_samples = 1000
 
soil_moisture = np.random.normal(30, 10, n_samples)       # percentage
temperature = np.random.normal(28, 5, n_samples)          # Celsius
humidity = np.random.normal(60, 10, n_samples)            # percentage
sunlight = np.random.normal(700, 150, n_samples)          # lux
 
# Simulate labels: water needed (1) or not (0)
# If soil is dry and temp is high or sunlight is strong, watering is needed
labels = ((soil_moisture < 25) & ((temperature > 30) | (sunlight > 800))).astype(int)
 
# Feature matrix
X = np.stack([soil_moisture, temperature, humidity, sunlight], axis=1)
y = labels
 
# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
 
# Build a simple binary classifier
model = models.Sequential([
    layers.Input(shape=(4,)),
    layers.Dense(32, activation='relu'),
    layers.Dense(16, activation='relu'),
    layers.Dense(1, activation='sigmoid')  # Binary output
])
 
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=15, batch_size=32, verbose=0)
 
# Evaluate
loss, acc = model.evaluate(X_test, y_test)
print(f"✅ Smart Irrigation Model Accuracy: {acc:.4f}")
 
# Predict water needs for next 5 samples
preds = (model.predict(X_test[:5]) > 0.5).astype(int).flatten()
for i in range(5):
    print(f"Sample {i+1}: Water Needed? {'Yes' if preds[i] else 'No'} (Actual: {'Yes' if y_test[i] else 'No'})")