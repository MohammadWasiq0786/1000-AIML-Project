"""
Project 816. Quality Control System

A quality control system monitors product characteristics to ensure they meet predefined standards. A simple yet powerful tool is the control chart, used to detect process variation over time. We'll simulate product measurements and plot a control chart to detect any values outside acceptable limits.

This code simulates quality measurements, calculates control limits, and highlights any out-of-control samples—key for detecting manufacturing or process issues early.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
 
# Simulated quality measurement data (e.g., product weight over time)
np.random.seed(42)
measurements = np.random.normal(loc=100, scale=2, size=30)  # 30 samples with mean=100g, std=2g
 
# Calculate control limits (±3 standard deviations from mean)
mean_val = np.mean(measurements)
std_val = np.std(measurements)
ucl = mean_val + 3 * std_val  # Upper Control Limit
lcl = mean_val - 3 * std_val  # Lower Control Limit
 
# Create DataFrame for plotting
df = pd.DataFrame({'Sample': np.arange(1, 31), 'Measurement': measurements})
 
# Plot control chart
plt.figure(figsize=(10, 5))
plt.plot(df['Sample'], df['Measurement'], marker='o', label='Measurements')
plt.axhline(mean_val, color='green', linestyle='-', label='Mean')
plt.axhline(ucl, color='red', linestyle='--', label='UCL (Upper Control Limit)')
plt.axhline(lcl, color='red', linestyle='--', label='LCL (Lower Control Limit)')
plt.fill_between(df['Sample'], lcl, ucl, color='lightgrey', alpha=0.3)
plt.title('Quality Control Chart')
plt.xlabel('Sample Number')
plt.ylabel('Measurement Value')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
 
# Identify out-of-control points
outliers = df[(df['Measurement'] > ucl) | (df['Measurement'] < lcl)]
print("Out-of-Control Samples:")
print(outliers)