"""
Project 292. Dynamic time warping implementation
Description:
Dynamic Time Warping (DTW) is a powerful algorithm for measuring similarity between two time series that may be misaligned or have different lengths/speeds. It’s used in:

Speech recognition

Activity tracking

Sensor pattern matching

We’ll implement DTW from scratch to compare two time series and visualize the optimal alignment path.

About:
✅ What It Does:
Compares two sequences with different timing

Finds the best alignment minimizing total distance

Visualizes the warping path to show correspondence

Can be applied to gesture comparison, ECG analysis, etc.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean
 
# 1. Generate two time series (one delayed version of the other)
t = np.linspace(0, 6.28, 100)
ts1 = np.sin(t)
ts2 = np.sin(t + 0.5) + np.random.normal(0, 0.05, size=len(t))
 
# 2. DTW implementation (dynamic programming)
def dtw(s1, s2):
    n, m = len(s1), len(s2)
    dtw_matrix = np.full((n+1, m+1), np.inf)
    dtw_matrix[0, 0] = 0
 
    for i in range(1, n+1):
        for j in range(1, m+1):
            cost = euclidean(s1[i-1], s2[j-1])
            dtw_matrix[i, j] = cost + min(
                dtw_matrix[i-1, j],     # insertion
                dtw_matrix[i, j-1],     # deletion
                dtw_matrix[i-1, j-1]    # match
            )
 
    # Backtracking to get the path
    path = []
    i, j = n, m
    while i > 0 and j > 0:
        path.append((i-1, j-1))
        i, j = min(
            [(i-1, j), (i, j-1), (i-1, j-1)],
            key=lambda x: dtw_matrix[x[0]+1, x[1]+1]
        )
    path.reverse()
    return dtw_matrix[n, m], path
 
# 3. Run DTW
distance, path = dtw(ts1.reshape(-1, 1), ts2.reshape(-1, 1))
print(f"DTW distance: {distance:.4f}")
 
# 4. Plot warping path
plt.figure(figsize=(8, 4))
for (i, j) in path[::5]:  # plot every 5th connection for clarity
    plt.plot([i, j], [ts1[i], ts2[j]], color='gray', linewidth=0.5)
plt.plot(ts1, label="Series 1")
plt.plot(ts2, label="Series 2")
plt.title("Dynamic Time Warping Alignment")
plt.legend()
plt.grid(True)
plt.show()