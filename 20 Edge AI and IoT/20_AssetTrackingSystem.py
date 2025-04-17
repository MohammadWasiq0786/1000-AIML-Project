"""
Project 780: Asset Tracking System
Description
An asset tracking system monitors the location and status of valuable assets (e.g., tools, containers, vehicles) in real-time using sensors like GPS, BLE, RFID, or accelerometers. In this simulation, we'll create synthetic location and movement data, and build a simple classifier to detect “Lost” vs. “Normal” status based on patterns such as inactivity, signal loss, or abnormal movement.

This kind of system is applicable in logistics, fleet management, and warehousing — and can be integrated with real GPS/BLE data on microcontrollers or edge gateways.
"""


import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
 
# Simulate asset sensor features: last known location variance, signal strength, days inactive, motion count
np.random.seed(42)
n_samples = 1000
 
location_variance = np.random.normal(5, 2, n_samples)           # low variance → stuck/lost
signal_strength = np.random.normal(70, 10, n_samples)           # RSSI in dBm
days_inactive = np.random.randint(0, 30, n_samples)             # higher = suspicious
motion_count = np.random.poisson(3, n_samples)                  # how often asset moves
 
# Create status labels (0 = normal, 1 = lost/suspicious)
labels = ((location_variance < 3) & (signal_strength < 60) & (days_inactive > 10)).astype(int)
 
# Feature matrix and labels
X = np.stack([location_variance, signal_strength, days_inactive, motion_count], axis=1)
y = labels
 
# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
 
# Build a simple classifier
model = models.Sequential([
    layers.Input(shape=(4,)),
    layers.Dense(32, activation='relu'),
    layers.Dense(16, activation='relu'),
    layers.Dense(1, activation='sigmoid')  # Binary: normal or lost
])
 
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=15, batch_size=32, verbose=0)
 
# Evaluate model
loss, acc = model.evaluate(X_test, y_test)
print(f"✅ Asset Tracking Model Accuracy: {acc:.4f}")
 
# Predict sample asset statuses
preds = (model.predict(X_test[:5]) > 0.5).astype(int).flatten()
for i in range(5):
    print(f"Asset {i+1}: Status = {'LOST' if preds[i] else 'NORMAL'} (Actual: {'LOST' if y_test[i] else 'NORMAL'})")