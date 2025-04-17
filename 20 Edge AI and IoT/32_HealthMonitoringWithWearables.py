"""
Project 792: Health Monitoring with Wearables
Description
Health monitoring systems in wearables collect continuous data such as heart rate, body temperature, and movement to detect abnormalities and support early diagnosis. In this project, we simulate physiological data and build a binary classification model to detect potential health alerts, such as signs of fatigue, dehydration, or elevated stress.

This model can power real-time alerts in fitness bands, smartwatches, or medical-grade wearables, and can be deployed using TensorFlow Lite for Microcontrollers (TFLM) on low-power chips.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
 
# Simulate wearable sensor data: heart rate (bpm), skin temperature (°C), activity level, blood oxygen (%)
np.random.seed(42)
n_samples = 1000
 
heart_rate = np.random.normal(75, 10, n_samples)
skin_temp = np.random.normal(36.5, 0.5, n_samples)
activity_level = np.random.normal(0.5, 0.2, n_samples)  # normalized from accelerometer
spo2 = np.random.normal(98, 1, n_samples)
 
# Label: 1 = alert if abnormal vitals (e.g., high HR, low SpO2, high temp with inactivity)
alert = ((heart_rate > 90) & (spo2 < 95) & (activity_level < 0.3) | (skin_temp > 37.5)).astype(int)
 
# Combine features
X = np.stack([heart_rate, skin_temp, activity_level, spo2], axis=1)
y = alert
 
# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
 
# Build health monitoring model
model = models.Sequential([
    layers.Input(shape=(4,)),
    layers.Dense(32, activation='relu'),
    layers.Dense(16, activation='relu'),
    layers.Dense(1, activation='sigmoid')  # Binary output: alert or normal
])
 
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=15, batch_size=32, verbose=0)
 
# Evaluate model
loss, acc = model.evaluate(X_test, y_test)
print(f"✅ Health Monitoring Model Accuracy: {acc:.4f}")
 
# Predict status for 5 individuals
preds = (model.predict(X_test[:5]) > 0.5).astype(int).flatten()
for i in range(5):
    print(f"User {i+1}: {'⚠️ Alert' if preds[i] else '✅ Normal'} (Actual: {'⚠️ Alert' if y_test[i] else '✅ Normal'})")