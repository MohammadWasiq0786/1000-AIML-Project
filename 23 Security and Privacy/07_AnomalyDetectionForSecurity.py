"""
Project 887. Anomaly Detection for Security

Anomaly detection for security identifies unexpected system behavior, such as unusual network usage, system calls, or user actions, which could indicate intrusions or internal threats. In this project, we simulate security log data and apply unsupervised anomaly detection using IsolationForest.

This model flags system sessions that deviate from normal patterns—ideal for early detection of compromise, insider threats, or resource misuse. For more complex cases, use:

Multivariate time-series anomaly detection

Autoencoders for feature compression and error reconstruction

Streaming anomaly detection with Apache Kafka or Flink
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
 
# Simulated security log features for system activity
data = {
    'CPU_Usage': [15, 20, 18, 17, 90, 22, 19, 95, 16, 21],  # % usage
    'Memory_Usage': [40, 45, 42, 43, 95, 41, 44, 97, 42, 46],  # % usage
    'NumProcesses': [60, 62, 59, 61, 150, 58, 60, 155, 63, 64]  # number of running processes
}
 
df = pd.DataFrame(data)
 
# Apply Isolation Forest for anomaly detection
model = IsolationForest(contamination=0.2, random_state=42)
df['Anomaly'] = model.fit_predict(df)
 
# Convert result: -1 = anomaly, 1 = normal
df['Anomaly'] = df['Anomaly'].map({1: 0, -1: 1})
 
# Show detected anomalies
print("Detected Anomalies in System Activity:")
print(df[df['Anomaly'] == 1])
 
# Visualize CPU vs Memory usage and highlight anomalies
plt.figure(figsize=(8, 5))
plt.scatter(df['CPU_Usage'], df['Memory_Usage'], c=df['Anomaly'], cmap='coolwarm', s=100, edgecolors='k')
plt.xlabel('CPU Usage (%)')
plt.ylabel('Memory Usage (%)')
plt.title('Security Anomaly Detection')
plt.grid(True)
plt.tight_layout()
plt.show()