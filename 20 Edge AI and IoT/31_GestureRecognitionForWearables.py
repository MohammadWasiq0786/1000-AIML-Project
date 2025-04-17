"""
Project 791: Gesture Recognition for Wearables
Description
Gesture recognition enables wearables like smartwatches or fitness bands to detect hand or body movements (e.g., swipe, shake, wave) using built-in motion sensors like accelerometers and gyroscopes. In this project, we simulate multiaxis motion data and build a 1D CNN classifier to recognize different gestures based on time-series input.

This type of model is perfect for on-device inference using TensorFlow Lite, enabling intuitive controls like tap-to-skip, flick-to-scroll, or shake-to-dismiss on wearables, AR glasses, or IoT wristbands.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
 
# Simulate gestures: swipe, shake, wave — using 3-axis accelerometer data over time
np.random.seed(42)
n_samples = 900
time_steps = 50  # readings per gesture
axes = 3         # x, y, z accelerometer
 
# Generate synthetic gesture patterns
def generate_gesture(pattern, label, count):
    base = {
        "swipe": [0.8, 0.1, 0.0],
        "shake": [0.5, 0.5, 0.5],
        "wave": [0.1, 0.8, 0.2]
    }[pattern]
    data = np.random.normal(loc=base, scale=0.2, size=(count, time_steps, axes))
    labels = np.full((count,), label)
    return data, labels
 
swipe_X, swipe_y = generate_gesture("swipe", "swipe", 300)
shake_X, shake_y = generate_gesture("shake", "shake", 300)
wave_X, wave_y = generate_gesture("wave", "wave", 300)
 
# Combine data
X = np.vstack([swipe_X, shake_X, wave_X])
y = np.concatenate([swipe_y, shake_y, wave_y])
 
# Encode labels
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)
 
# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
 
# Build 1D CNN model
model = models.Sequential([
    layers.Conv1D(32, 3, activation='relu', input_shape=(time_steps, axes)),
    layers.MaxPooling1D(pool_size=2),
    layers.Conv1D(64, 3, activation='relu'),
    layers.GlobalMaxPooling1D(),
    layers.Dense(64, activation='relu'),
    layers.Dense(3, activation='softmax')  # 3 gesture classes
])
 
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=15, batch_size=32, verbose=0)
 
# Evaluate model
loss, acc = model.evaluate(X_test, y_test)
print(f"✅ Gesture Recognition Accuracy: {acc:.4f}")
 
# Predict 5 sample gestures
preds = np.argmax(model.predict(X_test[:5]), axis=1)
for i in range(5):
    print(f"Sample {i+1}: Predicted = {encoder.classes_[preds[i]]}, Actual = {encoder.classes_[y_test[i]]}")