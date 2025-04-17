"""
Project 494: Customer Segmentation for Financial Services
Description:
Customer segmentation in financial services helps businesses categorize customers into different groups based on their behaviors, preferences, and financial activities. These segments can then be targeted with personalized offers, products, and services. In this project, we'll use K-means clustering to segment customers based on their financial features like account balance, number of transactions, and loan amount.

For real-world systems:

Use customer data from banks, credit card companies, or financial institutions to segment customers for marketing or service customization.

✅ What It Does:
Simulates customer data based on features like account balance, number of transactions, loan amount, and age.

Uses K-means clustering to segment customers into 4 groups based on the selected features.

Visualizes the customer segments using a scatter plot with color coding.

Summarizes the average values for each feature in each segment.

Key Extensions and Customizations:
Real-world datasets: Use actual customer data from banks or financial service providers.

Optimize clusters: Use methods like the Elbow Method to determine the optimal number of clusters.

Advanced segmentation: Use hierarchical clustering or DBSCAN for more flexibility in identifying customer segments.

Targeted marketing: Based on segments, create personalized marketing strategies or product offerings.
"""


import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
 
# 1. Simulate customer data
np.random.seed(42)
data = {
    'account_balance': np.random.normal(5000, 1500, 1000),  # USD
    'num_transactions': np.random.randint(1, 20, 1000),
    'loan_amount': np.random.normal(20000, 5000, 1000),  # USD
    'age': np.random.randint(18, 70, 1000)
}
 
df = pd.DataFrame(data)
 
# 2. Preprocessing
X = df[['account_balance', 'num_transactions', 'loan_amount', 'age']]
 
# 3. Scaling the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
 
# 4. Apply K-Means clustering
kmeans = KMeans(n_clusters=4, random_state=42)  # Assume 4 segments
df['Segment'] = kmeans.fit_predict(X_scaled)
 
# 5. Visualize the clusters
plt.figure(figsize=(10, 6))
plt.scatter(df['account_balance'], df['loan_amount'], c=df['Segment'], cmap='viridis', alpha=0.6)
plt.title('Customer Segmentation')
plt.xlabel('Account Balance (USD)')
plt.ylabel('Loan Amount (USD)')
plt.colorbar(label='Segment')
plt.show()
 
# 6. Show summary of each segment
segment_summary = df.groupby('Segment').mean()
print("Segment Summary (Average Features per Segment):\n")
print(segment_summary)