"""
Project 305. Time series segmentation
Description:
Time series segmentation involves dividing a time series into distinct regions where statistical properties (like mean or variance) change. This is useful for:

Activity recognition

Change point detection

Structural pattern analysis

In this project, we'll apply Kernel Change Point Detection to segment a series into regions of similar behavior.

About:
✅ What It Does:
Simulates a signal with 3 distinct statistical segments

Uses Kernel Change Point Detection from ruptures to detect transition points

Visually marks segment boundaries for easy interpretation

Ideal for detecting state transitions in real-world sensor or system data
"""

# Install if needed: pip install ruptures
 
import numpy as np
import matplotlib.pyplot as plt
import ruptures as rpt
 
# 1. Generate a piecewise signal with 3 segments
np.random.seed(42)
n_samples = 500
signal = np.concatenate([
    np.random.normal(loc=1.0, scale=0.3, size=150),
    np.random.normal(loc=3.0, scale=0.3, size=200),
    np.random.normal(loc=0.5, scale=0.3, size=150)
])
 
# 2. Apply change point detection
model = rpt.KernelCPD(kernel="linear").fit(signal.reshape(-1, 1))
change_points = model.predict(n_bkps=2)  # predict 2 change points
 
# 3. Plot results
plt.figure(figsize=(10, 4))
plt.plot(signal, label="Signal")
for cp in change_points:
    plt.axvline(cp, color='red', linestyle='--', label="Change Point" if cp == change_points[0] else "")
plt.title("Time Series Segmentation – Change Point Detection")
plt.legend()
plt.grid(True)
plt.show()