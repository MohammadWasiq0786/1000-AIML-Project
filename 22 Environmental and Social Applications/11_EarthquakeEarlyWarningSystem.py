"""
Project 851: Earthquake Early Warning System
Description
An Earthquake Early Warning System (EEWS) analyzes seismic signals in real time to detect earthquakes and issue alerts before destructive waves arrive. In this project, we simulate seismic sensor readings and build a binary classification model to determine whether a seismic signal indicates an impending earthquake.

✅ This system mimics real-world EEWS like:

Japan’s JMA EEW

USGS ShakeAlert

Can be integrated with seismic IoT sensors or accelerometers in smart buildings
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
 
# Simulate seismic sensor readings
np.random.seed(42)
n_samples = 1000
 
p_wave_amplitude = np.random.normal(0.3, 0.2, n_samples)    # primary wave
s_wave_amplitude = np.random.normal(0.5, 0.3, n_samples)    # secondary wave
ground_acceleration = np.random.normal(0.05, 0.02, n_samples)  # g-force
event_duration = np.random.normal(10, 3, n_samples)         # seconds
depth = np.random.normal(10, 5, n_samples)                  # km
 
# Label: Earthquake likely if p-wave & s-wave high, with high acceleration and shallow depth
earthquake = (
    (p_wave_amplitude > 0.4) &
    (s_wave_amplitude > 0.6) &
    (ground_acceleration > 0.06) &
    (depth < 15)
).astype(int)
 
# Feature matrix
X = np.stack([p_wave_amplitude, s_wave_amplitude, ground_acceleration, event_duration, depth], axis=1)
y = earthquake
 
# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
 
# Build a binary classifier
model = models.Sequential([
    layers.Input(shape=(5,)),
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(1, activation='sigmoid')  # Output: Earthquake? (Yes/No)
])
 
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=15, batch_size=32, verbose=0)
 
# Evaluate performance
loss, acc = model.evaluate(X_test, y_test)
print(f"✅ Earthquake Detection Accuracy: {acc:.4f}")
 
# Predict for 5 recent events
preds = (model.predict(X_test[:5]) > 0.5).astype(int).flatten()
for i in range(5):
    print(f"Event {i+1}: {'🌐 Earthquake Likely' if preds[i] else '✅ No Threat'} (Actual: {'🌐' if y_test[i] else '✅'})")