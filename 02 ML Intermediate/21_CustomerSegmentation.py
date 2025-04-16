"""
Project 61. Customer segmentation tool
Description:
A customer segmentation tool groups customers based on shared characteristics such as behavior, demographics, or purchase history. This enables targeted marketing and personalized experiences. In this project, we use K-Means clustering on synthetic customer data to identify distinct customer segments.

About:
🧠 What This Project Demonstrates:
Groups customers into segments using K-Means clustering

Preprocesses data with standard scaling

Visualizes high-dimensional clusters with PCA
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
 
# Simulated customer data
np.random.seed(42)
data = {
    'Age': np.random.randint(20, 65, 100),
    'AnnualIncome': np.random.randint(20000, 120000, 100),
    'SpendingScore': np.random.randint(1, 100, 100)  # e.g., loyalty score or purchase frequency
}
df = pd.DataFrame(data)
 
# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)
 
# Use KMeans for clustering
k = 4  # Number of customer segments
model = KMeans(n_clusters=k, random_state=42)
df['Segment'] = model.fit_predict(X_scaled)
 
# Optional: Reduce dimensions for visualization
pca = PCA(n_components=2)
pca_features = pca.fit_transform(X_scaled)
 
# Plot customer segments
plt.figure(figsize=(8, 5))
for cluster in range(k):
    plt.scatter(pca_features[df['Segment'] == cluster, 0], pca_features[df['Segment'] == cluster, 1], label=f'Segment {cluster}')
plt.title('Customer Segmentation (K-Means + PCA)')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()