"""
Project 304. Time series clustering
Description:
Time series clustering groups sequences based on similarity in shape, trend, or behavior. It’s useful in:

Customer segmentation

Sensor data profiling

Pattern discovery

In this project, we’ll:

Generate synthetic time series groups

Use Dynamic Time Warping (DTW) for distance measurement

Apply KMeans clustering with a DTW distance matrix

About:
✅ What It Does:
Creates three different shape-based groups of time series

Uses DTW to compare sequences regardless of temporal shifts

Clusters the sequences using KMeans with DTW

Visualizes groups with distinct trends or shapes
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from tslearn.metrics import cdist_dtw
from tslearn.clustering import TimeSeriesKMeans
 
# 1. Generate synthetic time series (3 groups)
np.random.seed(0)
n_series = 60
length = 50
 
group1 = np.sin(np.linspace(0, 3*np.pi, length)) + 0.1 * np.random.randn(n_series//3, length)
group2 = np.cos(np.linspace(0, 3*np.pi, length)) + 0.1 * np.random.randn(n_series//3, length)
group3 = np.linspace(0, 1, length) + 0.1 * np.random.randn(n_series//3, length)
data = np.concatenate([group1, group2, group3])
data = data[:, :, np.newaxis]  # shape: (n_series, length, 1)
 
# 2. Cluster using DTW
model = TimeSeriesKMeans(n_clusters=3, metric="dtw", max_iter=10, random_state=42)
labels = model.fit_predict(data)
 
# 3. Plot clusters
plt.figure(figsize=(12, 4))
for cluster in range(3):
    plt.subplot(1, 3, cluster + 1)
    for series in data[labels == cluster]:
        plt.plot(series.ravel(), alpha=0.5)
    plt.title(f"Cluster {cluster+1}")
    plt.grid(True)
plt.suptitle("Time Series Clustering using DTW + KMeans")
plt.tight_layout()
plt.show()