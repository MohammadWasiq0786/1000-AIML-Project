"""
Project 777: Industrial IoT Fault Detection
Description
Industrial IoT (IIoT) fault detection systems monitor machine parameters like vibration, temperature, pressure, or current to detect anomalies or faults in real-time. Early detection reduces downtime and maintenance costs. We'll simulate multivariate sensor data and use a binary classification model to identify fault conditions.

This simulation can be extended with real-time streaming data and deployed to edge devices like Raspberry Pi + accelerometer (ADXL345) or industrial gateways using TensorFlow Lite.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
 
# Simulate sensor readings: vibration, temperature, pressure, motor current
np.random.seed(42)
n_samples = 1200
 
vibration = np.random.normal(0.3, 0.1, n_samples)
temperature = np.random.normal(60, 5, n_samples)
pressure = np.random.normal(30, 3, n_samples)
current = np.random.normal(15, 2, n_samples)
 
# Simulate fault labels
# High vibration + high temperature or low pressure → fault
faults = ((vibration > 0.4) & ((temperature > 65) | (pressure < 28))).astype(int)
 
# Feature matrix and labels
X = np.stack([vibration, temperature, pressure, current], axis=1)
y = faults
 
# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
 
# Build binary classifier
model = models.Sequential([
    layers.Input(shape=(4,)),
    layers.Dense(32, activation='relu'),
    layers.Dense(16, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])
 
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=15, batch_size=32, verbose=0)
 
# Evaluate performance
loss, acc = model.evaluate(X_test, y_test)
print(f"✅ Industrial Fault Detection Model Accuracy: {acc:.4f}")
 
# Predict sample results
preds = (model.predict(X_test[:5]) > 0.5).astype(int).flatten()
for i in range(5):
    print(f"Sample {i+1}: Fault Detected? {'Yes' if preds[i] else 'No'} (Actual: {'Yes' if y_test[i] else 'No'})")