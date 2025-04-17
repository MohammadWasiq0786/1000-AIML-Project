"""
Project 786: Parking Space Detection
Description
Smart parking systems detect available parking spaces using sensors or camera feeds and guide vehicles accordingly. This project simulates sensor data from parking lots (e.g., ultrasonic or image-based sensors) and uses a binary classification model to predict whether a spot is occupied or vacant.

This can be adapted for real-time deployment on devices like Raspberry Pi + ultrasonic sensor, or integrated into camera-based detection systems using edge vision models.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
 
# Simulate features: distance reading (from ultrasonic), lighting level (lux), motion (1 if car just entered), time of day
np.random.seed(42)
n_samples = 1000
 
# Typical: occupied = short distance + lower light (car casts shadow), maybe recent motion
distance = np.random.normal(1.0, 0.5, n_samples)  # meters, low if car present
lighting = np.random.normal(300, 100, n_samples)  # lux
motion_detected = np.random.choice([0, 1], n_samples, p=[0.85, 0.15])
hour = np.random.randint(0, 24, n_samples)
 
# Labels: 1 = occupied
occupied = ((distance < 1.2) & (lighting < 250)).astype(int)
 
# Feature matrix and labels
X = np.stack([distance, lighting, motion_detected, hour], axis=1)
y = occupied
 
# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
 
# Build classifier
model = models.Sequential([
    layers.Input(shape=(4,)),
    layers.Dense(32, activation='relu'),
    layers.Dense(16, activation='relu'),
    layers.Dense(1, activation='sigmoid')  # Binary: occupied or not
])
 
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=15, batch_size=32, verbose=0)
 
# Evaluate
loss, acc = model.evaluate(X_test, y_test)
print(f"✅ Parking Spot Detection Accuracy: {acc:.4f}")
 
# Predict availability for 5 spots
preds = (model.predict(X_test[:5]) > 0.5).astype(int).flatten()
for i in range(5):
    print(f"Spot {i+1}: {'Occupied' if preds[i] else 'Vacant'} (Actual: {'Occupied' if y_test[i] else 'Vacant'})")