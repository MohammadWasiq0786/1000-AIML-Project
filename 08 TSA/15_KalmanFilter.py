"""
Project 295. Kalman filter implementation
Description:
The Kalman Filter is a powerful algorithm used to estimate the hidden state of a dynamic system from a series of noisy measurements. It’s widely used in:

Signal smoothing

Object tracking

Financial modeling

Sensor fusion

We’ll apply a basic 1D Kalman Filter to track a noisy time series and compare it with the raw data.

About:
✅ What It Does:
Simulates a noisy linear signal

Applies a 1D Kalman Filter to estimate the true value

Produces a smooth, adaptive estimate with low lag

Easily extendable to multi-dimensional or nonlinear systems
"""

import numpy as np
import matplotlib.pyplot as plt
 
# 1. Generate noisy observations around a true signal
np.random.seed(42)
n = 100
true_value = np.linspace(0, 10, n)
observations = true_value + np.random.normal(0, 1.0, size=n)  # noisy measurements
 
# 2. Basic Kalman Filter implementation
def kalman_filter(z, Q=1e-5, R=1.0):
    n = len(z)
    x_est = np.zeros(n)  # filtered estimate
    P = 1.0              # initial estimate error
    x = z[0]             # initial state estimate
 
    for i in range(n):
        # Prediction step
        x_pred = x
        P_pred = P + Q
 
        # Update step
        K = P_pred / (P_pred + R)
        x = x_pred + K * (z[i] - x_pred)
        P = (1 - K) * P_pred
 
        x_est[i] = x
 
    return x_est
 
# 3. Apply Kalman filter
smoothed = kalman_filter(observations)
 
# 4. Plot original vs filtered
plt.figure(figsize=(10, 4))
plt.plot(observations, label="Noisy Observations", alpha=0.6)
plt.plot(true_value, label="True Value", linestyle="--")
plt.plot(smoothed, label="Kalman Filter Estimate", linewidth=2)
plt.title("Kalman Filter – Signal Smoothing")
plt.xlabel("Time")
plt.ylabel("Value")
plt.legend()
plt.grid(True)
plt.show()