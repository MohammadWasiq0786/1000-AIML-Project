"""
Project 881. Network Intrusion Detection

A Network Intrusion Detection System (NIDS) monitors network traffic for suspicious activity or policy violations. In this project, we simulate network flow data and build a binary classification model to detect whether a connection is normal or an intrusion.

This model uses basic network session features to classify traffic as normal or malicious. In real-world applications, models are trained on datasets like NSL-KDD, CIC-IDS, or UNSW-NB15, and feature additional attributes like packet counts, flags, and timing.
"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
 
# Simulated network traffic dataset
# Each row represents a network connection session
data = {
    'Duration': [10, 300, 5, 200, 3, 250, 8, 100],
    'ProtocolType': [0, 1, 0, 1, 0, 1, 0, 1],  # e.g., 0 = TCP, 1 = UDP
    'BytesSent': [1000, 50000, 200, 40000, 150, 35000, 300, 12000],
    'BytesReceived': [800, 30000, 100, 25000, 100, 30000, 150, 10000],
    'Flag': [0, 1, 0, 1, 0, 1, 0, 1],  # Binary class: 0 = normal, 1 = intrusion
}
 
df = pd.DataFrame(data)
 
# Features and labels
X = df.drop('Flag', axis=1)
y = df['Flag']
 
# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
 
# Train a Random Forest classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
 
# Predict and evaluate
y_pred = model.predict(X_test)
print("Network Intrusion Detection Report:")
print(classification_report(y_test, y_pred))
 
# Predict on a new network session
new_connection = pd.DataFrame([{
    'Duration': 120,
    'ProtocolType': 0,
    'BytesSent': 10000,
    'BytesReceived': 8000
}])
intrusion_prob = model.predict_proba(new_connection)[0][1]
print(f"\nIntrusion Probability for new connection: {intrusion_prob:.2%}")