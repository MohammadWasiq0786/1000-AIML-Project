"""
Project 11. Anomaly detection using statistics
Description:
Anomaly detection identifies rare or unusual data points that differ significantly from the majority. Statistical methods like Z-score or IQR (Interquartile Range) are simple yet effective for detecting outliers in numerical data. In this project, we’ll use Z-score to find anomalies in a dataset representing daily transactions.

About:
What it does:

Uses Z-score to measure how far each data point is from the mean

Flags points that are unusually high or low as anomalies

Visualizes anomalies clearly with a simple line plot

You can also:

Use IQR-based detection (good for skewed data)

Extend to multivariate anomaly detection

Apply to real-world domains like fraud detection, sensor errors, or system monitoring
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import zscore
 
# Simulated dataset: daily transaction amounts in dollars
data = np.array([
    50, 52, 49, 51, 53, 48, 55, 54, 52, 49,
    51, 50, 500, 52, 47, 53, 49, 48, 700, 51
])
 
# Calculate Z-scores for each data point
z_scores = zscore(data)
 
# Define a threshold for anomaly detection (e.g., |Z| > 2.5)
threshold = 2.5
anomalies = np.where(np.abs(z_scores) > threshold)
 
# Print anomalies
print("Anomaly Detection using Z-Score:\n")
for i in anomalies[0]:
    print(f"Index {i}: Value = {data[i]}, Z-score = {z_scores[i]:.2f}")
 
# Visualize data with anomalies
plt.figure(figsize=(10, 5))
plt.plot(data, marker='o', label='Data Points')
plt.scatter(anomalies, data[anomalies], color='red', label='Anomalies')
plt.title("Transaction Anomaly Detection using Z-Score")
plt.xlabel("Day")
plt.ylabel("Transaction Amount ($)")
plt.legend()
plt.grid(True)
plt.show()