"""
Project 320. Online learning for time series
Description:
Online learning (or incremental learning) allows a model to update its parameters continuously as new data arrives, without retraining from scratch. This is useful for:

Real-time prediction

Adaptive models that improve as more data comes in

Memory-efficient models

In this project, we'll implement a simple online linear regression model that updates its coefficients as new time series data points arrive.

About:
✅ What It Does:
Simulates a time series with trend (linear data + noise)

Uses SGDRegressor to implement online linear regression (updating parameters incrementally)

Continuously updates the model with new data points arriving

Visualizes predictions being updated in real-time

This is a foundation for adaptive, real-time systems, such as:

Stock price prediction

Energy load forecasting

IoT sensor models
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDRegressor
 
# 1. Simulate a real-time time series with trend + noise
np.random.seed(42)
n = 1000
t = np.linspace(0, 10, n)
data = 2 * t + 1 + np.random.randn(n)  # Linear trend + noise
 
# 2. Initialize the online learning model (SGDRegressor)
model = SGDRegressor(max_iter=1, warm_start=True, penalty=None)
 
# 3. Prepare real-time plot
fig, ax = plt.subplots()
ax.set_xlim(0, 10)
ax.set_ylim(-5, 25)
line_data, = ax.plot(t, data, label="Data", alpha=0.5)
line_pred, = ax.plot([], [], label="Prediction", color='red', linewidth=2)
ax.set_title("Online Learning for Time Series")
ax.legend()
 
# 4. Online learning loop (simulate new data arriving)
predictions = []
for i in range(10, n):
    # Select the data point (current and previous values)
    X = t[i-10:i].reshape(-1, 1)  # Last 10 time steps
    y = data[i-10:i]  # Corresponding values
 
    # Fit the model incrementally
    model.fit(X, y)
    
    # Predict for the next step (online)
    prediction = model.predict(t[i].reshape(-1, 1))
 
    # Store prediction
    predictions.append(prediction[0])
 
    # Update plot
    line_pred.set_data(t[:i], predictions)
    plt.pause(0.01)
 
plt.show()