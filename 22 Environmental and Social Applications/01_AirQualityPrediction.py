"""
Project 841: Air Quality Prediction Model
Description
Air quality prediction helps identify pollution hotspots, issue public health alerts, and guide urban planning. In this project, we simulate environmental sensor data (e.g., temperature, humidity, PM2.5, NO2 levels) and build a regression model to predict the Air Quality Index (AQI), which is a standard measure of pollution severity.

This model can be trained with real-world data from:

UCI Air Quality Dataset

OpenAQ API

Local government pollution sensors
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
 
# Simulate sensor inputs: PM2.5, PM10, NO2, CO, temperature, humidity
np.random.seed(42)
n_samples = 1000
 
pm25 = np.random.normal(40, 10, n_samples)        # μg/m3
pm10 = np.random.normal(60, 15, n_samples)        # μg/m3
no2 = np.random.normal(30, 8, n_samples)          # ppb
co = np.random.normal(0.7, 0.2, n_samples)        # ppm
temperature = np.random.normal(25, 5, n_samples)  # °C
humidity = np.random.normal(60, 10, n_samples)    # %
 
# Simulate AQI using a weighted combination + noise
aqi = (0.4 * pm25 + 0.3 * pm10 + 0.15 * no2 + 0.1 * co * 100 + 
       0.02 * temperature - 0.01 * humidity + np.random.normal(0, 5, n_samples))
 
# Feature matrix
X = np.stack([pm25, pm10, no2, co, temperature, humidity], axis=1)
y = aqi
 
# Split into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
 
# Build regression model
model = models.Sequential([
    layers.Input(shape=(6,)),
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(1)  # Output: AQI value
])
 
model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=0)
 
# Evaluate
loss, mae = model.evaluate(X_test, y_test)
print(f"✅ Air Quality Model MAE: {mae:.2f} AQI units")
 
# Predict and plot comparison
predicted = model.predict(X_test[:50]).flatten()
actual = y_test[:50]
 
plt.figure(figsize=(10, 4))
plt.plot(actual, label="Actual AQI")
plt.plot(predicted, label="Predicted AQI")
plt.title("Air Quality Prediction")
plt.xlabel("Sample")
plt.ylabel("AQI")
plt.legend()
plt.grid(True)
plt.show()