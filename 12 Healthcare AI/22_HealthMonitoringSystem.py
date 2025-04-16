"""
Project 462. Health monitoring system
Description:
A Health Monitoring System continuously tracks user vitals such as heart rate, temperature, SpO2, or steps, and provides alerts when anomalies occur. In this project, we'll simulate real-time data streaming and use basic threshold-based logic to flag abnormal health conditions.

About:
✅ What It Does:
Simulates live data from health sensors.

Monitors for threshold violations.

Prints real-time alerts for abnormal vitals.

Can be extended to:

Integrate live wearable data via APIs

Send email/SMS notifications

Use machine learning models for personalized baselines and anomaly detection

For real-world systems:

Integrate with wearable APIs (e.g., Fitbit, Apple Health, Garmin)

Use platforms like Edge AI, Raspberry Pi, or IoT sensors
"""


import random
import time
 
# 1. Thresholds for alerting
THRESHOLDS = {
    "heart_rate": (60, 100),        # normal range bpm
    "temperature": (36.1, 37.5),    # Celsius
    "spo2": (95, 100)               # SpO2 percentage
}
 
# 2. Simulated sensor data generator
def get_sensor_data():
    return {
        "heart_rate": random.randint(50, 120),
        "temperature": round(random.uniform(35.5, 38.5), 1),
        "spo2": random.randint(90, 100)
    }
 
# 3. Monitoring loop with alerts
print("Starting Health Monitoring System...\nPress Ctrl+C to stop.\n")
try:
    while True:
        data = get_sensor_data()
        print("Current Vitals:", data)
 
        # Check for alerts
        alerts = []
        for key, value in data.items():
            low, high = THRESHOLDS[key]
            if not (low <= value <= high):
                alerts.append(f"Abnormal {key.replace('_', ' ').title()}: {value}")
 
        if alerts:
            print("ALERTS:")
            for alert in alerts:
                print(" -", alert)
        else:
            print("All vitals are within normal range.")
 
        print("-" * 50)
        time.sleep(2)  # simulate real-time data every 2 seconds
 
except KeyboardInterrupt:
    print("\nHealth Monitoring System stopped.")