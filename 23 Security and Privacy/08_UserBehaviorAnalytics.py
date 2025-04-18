"""
Project 888. User Behavior Analytics

User Behavior Analytics (UBA) tracks and analyzes user activity to detect abnormal patterns that might indicate insider threats, account compromise, or policy violations. In this project, we simulate user session logs and use unsupervised clustering (K-Means) to detect outliers in user behavior.

This project identifies abnormal users by grouping behavior and labeling outliers. You can enhance UBA with:

Time-series modeling (e.g., user actions over sessions)

Role-based behavior baselining

Integration with SIEM systems
"""

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
 
# Simulated user activity logs
data = {
    'LoginDuration': [5, 6, 4, 5, 60, 4, 5, 65, 4, 5],  # in minutes
    'FilesAccessed': [10, 12, 8, 11, 50, 9, 10, 55, 9, 11],  # number of files
    'FailedLogins': [0, 1, 0, 0, 5, 0, 1, 6, 0, 1]  # number of failed login attempts
}
 
df = pd.DataFrame(data)
 
# Normalize the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)
 
# Fit K-Means with 2 clusters (normal vs outlier)
kmeans = KMeans(n_clusters=2, random_state=42)
df['Cluster'] = kmeans.fit_predict(scaled_data)
 
# Assume the cluster with fewer members is the 'abnormal' one
counts = df['Cluster'].value_counts()
abnormal_cluster = counts.idxmin()
df['Anomaly'] = df['Cluster'].apply(lambda x: 1 if x == abnormal_cluster else 0)
 
# Display abnormal user behavior
print("Abnormal User Sessions:")
print(df[df['Anomaly'] == 1])
 
# Visualize clustering
plt.figure(figsize=(8, 5))
plt.scatter(df['LoginDuration'], df['FilesAccessed'], c=df['Anomaly'], cmap='coolwarm', s=100, edgecolors='k')
plt.xlabel('Login Duration (min)')
plt.ylabel('Files Accessed')
plt.title('User Behavior Analytics - Anomaly Detection')
plt.grid(True)
plt.tight_layout()
plt.show()