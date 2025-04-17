"""
Project 789: Crowd Monitoring System
Description
Crowd monitoring systems help track people density and movement in public spaces like malls, airports, or events. Using data from cameras, WiFi/BLE beacons, or infrared sensors, such systems can detect overcrowding, assist in safety alerts, and optimize flow. In this simulation, we build a classifier to predict crowd congestion levels (low, medium, high) using synthetic sensor features.

This model can be connected to real-time video analytics, BLE scanner logs, or thermal camera sensors and deployed on edge nodes for fast alerts during peak crowd conditions.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
 
# Simulate sensor inputs: people count, average dwell time (mins), WiFi signal overlap, thermal activity
np.random.seed(42)
n_samples = 1000
 
people_count = np.random.randint(10, 200, n_samples)
dwell_time = np.random.normal(5, 2, n_samples)
wifi_overlap = np.random.normal(0.6, 0.2, n_samples)  # proxy for density
thermal_readings = np.random.normal(30, 3, n_samples)  # crowd = higher heat
 
# Label density: low (<50), medium (50–120), high (>120) based on people count
density = np.where(people_count < 50, 'low',
           np.where(people_count < 120, 'medium', 'high'))
 
# Encode labels
encoder = LabelEncoder()
y = encoder.fit_transform(density)
 
# Feature matrix
X = np.stack([people_count, dwell_time, wifi_overlap, thermal_readings], axis=1)
 
# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
 
# Build multi-class classifier
model = models.Sequential([
    layers.Input(shape=(4,)),
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(3, activation='softmax')  # 3 classes: low, medium, high
])
 
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=15, batch_size=32, verbose=0)
 
# Evaluate accuracy
loss, acc = model.evaluate(X_test, y_test)
print(f"✅ Crowd Monitoring Model Accuracy: {acc:.4f}")
 
# Predict crowd level for 5 samples
preds = np.argmax(model.predict(X_test[:5]), axis=1)
for i in range(5):
    print(f"Zone {i+1}: Predicted = {encoder.classes_[preds[i]]}, Actual = {encoder.classes_[y_test[i]]}")