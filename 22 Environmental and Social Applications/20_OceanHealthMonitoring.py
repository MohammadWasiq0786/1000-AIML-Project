"""
Project 860: Ocean Health Monitoring
Description
Monitoring ocean health involves tracking variables like sea surface temperature, chlorophyll concentration, pH levels, and dissolved oxygen. This helps detect coral bleaching, algal blooms, or dead zones. In this project, we simulate oceanographic data and build a multi-class classifier to assess the health status of ocean regions (Healthy, Moderate Risk, Critical).

✅ Use Cases:

Satellite + buoy sensor fusion

Marine conservation dashboards

Early warning for coral bleaching, red tides, or low-oxygen zones
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
 
# Simulate ocean sensor data
np.random.seed(42)
n_samples = 1000
 
sea_temp = np.random.normal(26, 2, n_samples)               # °C
chlorophyll = np.random.normal(1.5, 0.5, n_samples)         # mg/m³
ph_level = np.random.normal(8.1, 0.1, n_samples)            # pH
dissolved_oxygen = np.random.normal(6, 1, n_samples)        # mg/L
salinity = np.random.normal(35, 1, n_samples)               # PSU
 
# Label: 0 = Healthy, 1 = Moderate Risk, 2 = Critical
# Based on thresholds of temp, oxygen, and chlorophyll
health_status = np.where(
    (sea_temp > 28) | (chlorophyll > 2.5) | (dissolved_oxygen < 4.5), 2,  # Critical
    np.where((sea_temp > 27) | (chlorophyll > 2.0) | (dissolved_oxygen < 5.5), 1,  # Moderate Risk
    0)  # Healthy
)
 
# Stack features
X = np.stack([sea_temp, chlorophyll, ph_level, dissolved_oxygen, salinity], axis=1)
y = health_status
 
# Split into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
 
# Build multi-class classification model
model = models.Sequential([
    layers.Input(shape=(5,)),
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(3, activation='softmax')  # 3 ocean health classes
])
 
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=0)
 
# Evaluate performance
loss, acc = model.evaluate(X_test, y_test)
print(f"✅ Ocean Health Monitoring Accuracy: {acc:.4f}")
 
# Predict for 5 ocean regions
preds = np.argmax(model.predict(X_test[:5]), axis=1)
health_map = {0: "✅ Healthy", 1: "⚠️ Moderate Risk", 2: "🚨 Critical Condition"}
 
for i in range(5):
    print(f"Region {i+1}: Predicted = {health_map[preds[i]]}, Actual = {health_map[y_test[i]]}")