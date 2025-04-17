"""
Project 778: Predictive Maintenance System
Description
A predictive maintenance system forecasts equipment failures before they happen by analyzing sensor patterns over time. This helps reduce downtime, extend machinery life, and optimize maintenance schedules. We’ll simulate time-series sensor data and build a sequence classification model using LSTM to predict whether maintenance is needed.

This predictive maintenance system can be extended to work with real sensor streams and integrated into platforms like Edge Impulse, Azure IoT, or AWS Greengrass for real-time edge analytics.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
 
# Simulate sequential sensor data (e.g., vibration readings over time)
np.random.seed(42)
n_samples = 1000
time_steps = 50  # readings per session
features = 3     # vibration, temperature, pressure
 
# Generate normal operational patterns
normal_data = np.random.normal(loc=[0.3, 60, 30], scale=[0.05, 2, 1], size=(int(n_samples * 0.7), time_steps, features))
normal_labels = np.zeros((normal_data.shape[0],))
 
# Generate faulty patterns
faulty_data = np.random.normal(loc=[0.5, 70, 27], scale=[0.1, 3, 2], size=(int(n_samples * 0.3), time_steps, features))
faulty_labels = np.ones((faulty_data.shape[0],))
 
# Combine and shuffle
X = np.vstack([normal_data, faulty_data])
y = np.concatenate([normal_labels, faulty_labels])
 
# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
 
# Build LSTM sequence model
model = models.Sequential([
    layers.LSTM(32, input_shape=(time_steps, features)),
    layers.Dense(16, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])
 
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)
 
# Evaluate
loss, acc = model.evaluate(X_test, y_test)
print(f"✅ Predictive Maintenance Model Accuracy: {acc:.4f}")
 
# Predict maintenance needs
preds = (model.predict(X_test[:5]) > 0.5).astype(int).flatten()
for i in range(5):
    print(f"Machine {i+1}: Maintenance Required? {'Yes' if preds[i] else 'No'} (Actual: {'Yes' if y_test[i] else 'No'})")