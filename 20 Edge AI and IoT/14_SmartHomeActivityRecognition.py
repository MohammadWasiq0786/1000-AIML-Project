"""
Project 774: Smart Home Activity Recognition
Description
Smart home activity recognition involves detecting what a resident is doing (e.g., cooking, sleeping, walking) using sensor data from motion, door, temperature, and appliance sensors. This enables personalized automation and safety alerts. We'll simulate this using synthetic multivariate time-series data and build a classifier using a 1D CNN, which is efficient for edge deployment.

This model can be deployed on edge devices (like a Raspberry Pi or ESP32 with TensorFlow Lite) to recognize user activity in real-time based on sensor fusion inputs.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
 
# Simulate smart home sensor data for 3 activities
# Each sample has 100 time steps and 3 sensors (e.g., motion, appliance, door)
def generate_data(activity, label, num_samples=200):
    base = {
        "cooking": [1.0, 0.3, 0.8],
        "sleeping": [0.1, 0.05, 0.0],
        "walking": [0.6, 0.2, 0.5]
    }[activity]
    data = np.random.normal(loc=base, scale=0.1, size=(num_samples, 100, 3))
    labels = np.full((num_samples,), label)
    return data, labels
 
# Generate dataset
cooking_X, cooking_y = generate_data("cooking", "cooking")
sleeping_X, sleeping_y = generate_data("sleeping", "sleeping")
walking_X, walking_y = generate_data("walking", "walking")
 
# Combine all data
X = np.vstack([cooking_X, sleeping_X, walking_X])
y = np.concatenate([cooking_y, sleeping_y, walking_y])
 
# Encode labels
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)
 
# Split into train/test sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
 
# Build 1D CNN model
model = models.Sequential([
    layers.Conv1D(32, kernel_size=3, activation='relu', input_shape=(100, 3)),
    layers.MaxPooling1D(pool_size=2),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(3, activation='softmax')  # 3 activities
])
 
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
 
# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)
 
# Evaluate the model
loss, acc = model.evaluate(X_test, y_test)
print(f"✅ Smart Home Activity Recognition Accuracy: {acc:.4f}")
 
# Predict a few samples
preds = model.predict(X_test[:5])
for i, pred in enumerate(preds):
    print(f"Sample {i+1}: Predicted = {encoder.classes_[np.argmax(pred)]}, True = {encoder.classes_[y_test[i]]}")