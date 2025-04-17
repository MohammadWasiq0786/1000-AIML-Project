"""
Project 785: Traffic Monitoring System
Description
A traffic monitoring system uses sensors or cameras to track vehicle flow, detect congestion, and optimize signal timing in real-time. In this project, we simulate traffic sensor data (vehicle count, speed, weather, time of day) and use a classifier model to predict traffic congestion status (congested vs. smooth).

This approach can be adapted for live feeds from camera-based CV systems, loop detectors, or GPS-enabled fleet data to inform smart traffic signals, routing apps, or city dashboards.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
 
# Simulate traffic sensor inputs: vehicle count per minute, average speed (km/h), weather condition (0=clear, 1=rain), time of day (0-23)
np.random.seed(42)
n_samples = 1000
 
vehicle_count = np.random.randint(10, 150, n_samples)
avg_speed = np.random.normal(50, 10, n_samples)
weather = np.random.choice([0, 1], n_samples, p=[0.7, 0.3])
hour = np.random.randint(0, 24, n_samples)
 
# Label: 1 = congested if high vehicle count and low speed, especially during peak hours or rain
congestion = ((vehicle_count > 100) & (avg_speed < 40) & ((hour >= 7) & (hour <= 10) | (hour >= 17) & (hour <= 20) | (weather == 1))).astype(int)
 
# Feature matrix and labels
X = np.stack([vehicle_count, avg_speed, weather, hour], axis=1)
y = congestion
 
# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
 
# Build traffic congestion classifier
model = models.Sequential([
    layers.Input(shape=(4,)),
    layers.Dense(32, activation='relu'),
    layers.Dense(16, activation='relu'),
    layers.Dense(1, activation='sigmoid')  # Binary classification
])
 
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=15, batch_size=32, verbose=0)
 
# Evaluate model
loss, acc = model.evaluate(X_test, y_test)
print(f"✅ Traffic Monitoring Model Accuracy: {acc:.4f}")
 
# Predict on new sensor data
preds = (model.predict(X_test[:5]) > 0.5).astype(int).flatten()
for i in range(5):
    print(f"Sample {i+1}: {'Congested' if preds[i] else 'Smooth'} (Actual: {'Congested' if y_test[i] else 'Smooth'})")