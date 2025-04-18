"""
Project 867: Traffic Optimization for Emissions Reduction
Description
Traffic congestion increases fuel usage and air pollution. In this project, we simulate traffic flow data and build a regression model to estimate emissions levels and identify optimal traffic signal timing or rerouting recommendations for reducing carbon output.

✅ This model powers:

Smart traffic signal timing

Urban carbon footprint analysis

Green navigation systems for vehicles and public buses
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
 
# Simulate traffic sensor data
np.random.seed(42)
n_samples = 1000
 
vehicle_count = np.random.poisson(50, n_samples)              # vehicles per intersection per cycle
avg_speed = np.random.normal(30, 10, n_samples)               # km/h
idle_time = np.random.normal(20, 10, n_samples)               # seconds
signal_cycle_time = np.random.normal(90, 20, n_samples)       # seconds
intersection_density = np.random.normal(5, 2, n_samples)      # intersections/km²
 
# Simulate emissions (kg CO₂ per intersection per hour)
emissions = (
    0.2 * vehicle_count +
    0.5 * (signal_cycle_time / 90) +
    0.3 * (idle_time / 60) -
    0.1 * avg_speed +
    0.2 * intersection_density +
    np.random.normal(0, 1, n_samples)
)
 
# Feature matrix
X = np.stack([vehicle_count, avg_speed, idle_time, signal_cycle_time, intersection_density], axis=1)
y = emissions
 
# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
 
# Build regression model
model = models.Sequential([
    layers.Input(shape=(5,)),
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(1)  # Output: CO₂ emissions
])
 
model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=0)
 
# Evaluate model
loss, mae = model.evaluate(X_test, y_test)
print(f"✅ Traffic Emissions Prediction MAE: {mae:.2f} kg CO₂/hr")
 
# Predict for 5 intersections
preds = model.predict(X_test[:5]).flatten()
for i in range(5):
    print(f"Intersection {i+1}: Predicted Emissions = {preds[i]:.2f} kg CO₂/hr (Actual: {y_test[i]:.2f})")