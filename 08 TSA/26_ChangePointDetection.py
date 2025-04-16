"""
Project 306. Change point detection
Description:
Change Point Detection identifies points in time where the statistical properties of a time series — like mean, variance, or distribution — change abruptly. This is essential in:

Fault detection

Financial trend shifts

Behavioral pattern changes

In this project, we'll detect multiple change points in a noisy time series using the ruptures library and the Pelt algorithm with a cost function.

About:
✅ What It Does:
Generates a time series with four segments, each with a different mean

Applies PELT (Pruned Exact Linear Time) algorithm for efficient change point detection

Visualizes detected points where distribution changes occur

Penalty term controls the sensitivity of change detection
"""

import numpy as np
import matplotlib.pyplot as plt
import ruptures as rpt
 
# 1. Generate a time series with multiple shifts in mean
np.random.seed(42)
signal = np.concatenate([
    np.random.normal(loc=2, scale=0.3, size=150),
    np.random.normal(loc=0, scale=0.3, size=100),
    np.random.normal(loc=1.5, scale=0.3, size=120),
    np.random.normal(loc=-1, scale=0.3, size=130)
])
 
# 2. Apply PELT change point detection with L2 cost
model = rpt.Pelt(model="l2").fit(signal)
change_points = model.predict(pen=10)  # penalty determines sensitivity
 
# 3. Plot the signal with detected change points
plt.figure(figsize=(10, 4))
plt.plot(signal, label="Signal")
for cp in change_points[:-1]:  # ignore the last point (end of signal)
    plt.axvline(cp, color='red', linestyle='--', label="Change Point" if cp == change_points[0] else "")
plt.title("Change Point Detection using PELT")
plt.xlabel("Time")
plt.ylabel("Value")
plt.legend()
plt.grid(True)
plt.show()