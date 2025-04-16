"""
Project 3. K-means clustering for customer segmentation
Description:
K-Means Clustering is an unsupervised learning algorithm used to group similar data points into k clusters based on feature similarity. In this project, we’ll apply K-Means to simulate customer data (e.g., annual income and spending score) to identify customer segments like "high spenders" or "budget-conscious" shoppers.

About:
This project segments customers based on their income and spending habits, helping businesses to target groups more effectively—like offering premium products to high-income, high-spending clusters.
"""

# Import required libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
 
# Sample data: (Annual Income in $1000s, Spending Score)
# This simulates basic customer behavior
X = np.array([
    [15, 39], [16, 81], [17, 6], [18, 77], [19, 40], [20, 76],
    [25, 50], [30, 60], [35, 80], [40, 20],
    [60, 85], [65, 70], [70, 60], [75, 50], [80, 30],
    [85, 90], [90, 70], [95, 40], [100, 20], [105, 10]
])
 
# Create KMeans model with k=3 clusters
kmeans = KMeans(n_clusters=3, random_state=42)
 
# Fit the model and predict clusters
y_kmeans = kmeans.fit_predict(X)
 
# Get cluster centers
centers = kmeans.cluster_centers_
 
# Plot the clusters with different colors
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, cmap='viridis', label='Customers')
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, marker='X', label='Centroids')
plt.xlabel("Annual Income ($1000s)")
plt.ylabel("Spending Score")
plt.title("Customer Segmentation using K-Means Clustering")
plt.legend()
plt.grid(True)
plt.show()
 
# Display final cluster centers
print("Cluster Centers (Annual Income, Spending Score):")
for i, center in enumerate(centers):
    print(f"Cluster {i + 1}: Income = {center[0]:.2f}, Spending Score = {center[1]:.2f}")