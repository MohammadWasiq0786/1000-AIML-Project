"""
Project 801. Customer Segmentation Analysis

This project involves segmenting customers into distinct groups based on purchasing behavior or demographic attributes. By clustering similar customers together, businesses can tailor marketing strategies, product recommendations, or loyalty programs more effectively. We'll use the K-Means clustering algorithm on a synthetic dataset containing customer features such as age, income, and spending score.

This code demonstrates how to group customers based on income and spending habits, helping businesses understand different customer profiles (e.g., high spenders, budget-conscious, etc.) for targeted actions.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
 
# Create a sample dataset simulating customer data
data = {
    'Age': [19, 35, 26, 27, 19, 27, 27, 32, 25, 35],
    'Annual_Income_k$': [15, 35, 35, 19, 27, 75, 45, 40, 60, 50],
    'Spending_Score': [39, 81, 6, 77, 40, 76, 20, 8, 40, 35]
}
 
# Convert to DataFrame
df = pd.DataFrame(data)
 
# Scale the features for better clustering performance
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)
 
# Apply KMeans clustering
# Here we choose 3 clusters arbitrarily (can use elbow method to choose optimal)
kmeans = KMeans(n_clusters=3, random_state=42)
df['Cluster'] = kmeans.fit_predict(scaled_data)
 
# Visualize the clusters
plt.figure(figsize=(8, 5))
plt.scatter(df['Annual_Income_k$'], df['Spending_Score'], 
            c=df['Cluster'], cmap='viridis', s=100)
 
# Plot cluster centers
centers = scaler.inverse_transform(kmeans.cluster_centers_)  # scale back for original values
plt.scatter(centers[:, 1], centers[:, 2], c='red', s=200, alpha=0.75, marker='X', label='Centroids')
 
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score')
plt.title('Customer Segmentation using K-Means')
plt.legend()
plt.grid(True)
plt.show()
 
# Display the segmented customer data
print(df)