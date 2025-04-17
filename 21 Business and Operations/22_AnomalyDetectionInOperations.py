"""
Project 822. Anomaly Detection in Operations

Anomaly detection helps identify unexpected behavior in operational data, such as spikes in process time, unexpected sensor values, or unusual demand. In this project, we simulate time-series data and use Isolation Forest, an unsupervised learning model, to detect anomalies without prior labeling.

This implementation uses Isolation Forest to detect unusual patterns in unlabelled operational data, useful for monitoring KPIs, detecting equipment drift, or spotting abnormal system behavior in real-time.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
 
# Simulated operational metric data (e.g., process time in seconds)
np.random.seed(42)
normal_data = np.random.normal(loc=50, scale=5, size=50)
anomalies = [80, 85, 90]  # Injected anomalies
data_values = np.concatenate([normal_data, anomalies])
 
df = pd.DataFrame({'MetricValue': data_values})
 
# Train an Isolation Forest model
model = IsolationForest(contamination=0.05, random_state=42)
df['Anomaly'] = model.fit_predict(df[['MetricValue']])
 
# Anomaly labels: -1 = anomaly, 1 = normal
df['Anomaly'] = df['Anomaly'].map({1: 0, -1: 1})
 
# Plotting results
plt.figure(figsize=(10, 5))
plt.plot(df.index, df['MetricValue'], marker='o', label='Metric Value')
plt.scatter(df.index[df['Anomaly'] == 1], 
            df['MetricValue'][df['Anomaly'] == 1], 
            color='red', label='Anomaly', zorder=5)
plt.title('Operational Metric Anomaly Detection')
plt.xlabel('Observation Index')
plt.ylabel('Metric Value')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
 
# Show detected anomalies
print("Detected Anomalies:")
print(df[df['Anomaly'] == 1])