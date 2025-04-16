"""
Project 297. State space models
Description:
State Space Models (SSMs) are a flexible framework combining:

Latent (hidden) states that evolve over time

Observations generated from those states
They generalize Kalman Filters and Hidden Markov Models, supporting both continuous and discrete systems.

In this project, we’ll build and apply a simple linear Gaussian state space model to smooth and predict a noisy signal.

About:
✅ What It Does:
Simulates a noisy seasonal signal with trend

Models it using a local linear trend + seasonal component

Extracts the hidden smoothed state

Forecasts future values using learned latent structure
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.kalman_filter import KalmanFilter
from statsmodels.tsa.statespace.structural import UnobservedComponents
 
# 1. Simulate noisy trend + seasonal time series
np.random.seed(42)
n = 200
time = np.arange(n)
trend = 0.05 * time
seasonal = 2 * np.sin(2 * np.pi * time / 12)
noise = np.random.normal(scale=0.5, size=n)
observed = trend + seasonal + noise
 
# 2. Define State Space Model (trend + seasonal)
model = UnobservedComponents(observed, level='local linear trend', seasonal=12)
result = model.fit(disp=False)
 
# 3. Get smoothed estimates and forecasts
smoothed = result.smoothed_state[0]  # level
forecast = result.forecast(steps=12)
 
# 4. Plot original, smoothed, and forecast
plt.figure(figsize=(12, 4))
plt.plot(observed, label="Observed")
plt.plot(smoothed, label="Smoothed Trend", linewidth=2)
plt.plot(np.arange(n, n + 12), forecast, label="Forecast", linestyle='--')
plt.title("State Space Model – Trend + Seasonal Smoothing")
plt.xlabel("Time")
plt.ylabel("Value")
plt.legend()
plt.grid(True)
plt.show()