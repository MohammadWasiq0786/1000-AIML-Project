"""
Project 833. Workplace Safety Monitoring

Workplace safety monitoring involves detecting and responding to unsafe conditions or incidents in real-time. In this project, we simulate sensor data (like temperature, gas level, and movement) and use threshold-based rules combined with anomaly detection to flag safety violations.

This hybrid model flags readings based on hardcoded thresholds and anomalies detected via Isolation Forest—useful for early warning systems in factories, mines, or labs.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
 
# Simulated workplace sensor data
data = {
    'Temperature': [22, 23, 24, 25, 80, 22, 23, 85, 24, 23],     # °C
    'GasLevel': [5, 4, 6, 5, 30, 4, 5, 35, 6, 5],                # ppm
    'Vibration': [0.2, 0.3, 0.2, 0.4, 1.5, 0.3, 0.3, 1.8, 0.2, 0.3]  # G-force
}
 
df = pd.DataFrame(data)
 
# Simple threshold rule-based monitoring
def check_thresholds(row):
    if row['Temperature'] > 60 or row['GasLevel'] > 25 or row['Vibration'] > 1.0:
        return 1  # Unsafe
    return 0  # Safe
 
df['UnsafeCondition'] = df.apply(check_thresholds, axis=1)
 
# Anomaly detection using Isolation Forest
model = IsolationForest(contamination=0.2, random_state=42)
df['Anomaly'] = model.fit_predict(df[['Temperature', 'GasLevel', 'Vibration']])
df['Anomaly'] = df['Anomaly'].map({1: 0, -1: 1})  # 1 = anomaly
 
# Final flag: any row marked as unsafe or anomalous
df['SafetyAlert'] = df[['UnsafeCondition', 'Anomaly']].max(axis=1)
 
# Show flagged safety alerts
print("Workplace Safety Alerts:")
print(df[df['SafetyAlert'] == 1])
 
# Visualize alerts
plt.figure(figsize=(10, 5))
plt.plot(df.index, df['Temperature'], label='Temperature (°C)', marker='o')
plt.plot(df.index, df['GasLevel'], label='Gas Level (ppm)', marker='x')
plt.plot(df.index, df['Vibration'], label='Vibration (G)', marker='s')
plt.scatter(df.index[df['SafetyAlert'] == 1], 
            df['Temperature'][df['SafetyAlert'] == 1], 
            color='red', label='Safety Alert', zorder=5)
plt.legend()
plt.title('Sensor Readings and Safety Alerts')
plt.xlabel('Time Index')
plt.grid(True)
plt.tight_layout()
plt.show()