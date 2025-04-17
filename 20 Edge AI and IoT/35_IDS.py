"""
Project 795: Intrusion Detection System (IDS)
Description
An Intrusion Detection System monitors network or physical activity to detect unauthorized access. In this project, we simulate a network-based IDS using synthetic traffic features like packet size, connection duration, and flags, and train a binary classifier to detect malicious vs. normal traffic — applicable for IoT security and smart building firewalls.

This system forms the AI core of edge security appliances, smart routers, or IoT firewall systems. For real-world use, models can be trained on datasets like NSL-KDD, UNSW-NB15, or CICIDS.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
 
# Simulate network traffic features: duration (s), bytes sent, bytes received, failed logins, suspicious flag count
np.random.seed(42)
n_samples = 1000
 
duration = np.random.exponential(scale=2.0, size=n_samples)
bytes_sent = np.random.normal(1000, 300, n_samples)
bytes_received = np.random.normal(1200, 400, n_samples)
failed_logins = np.random.poisson(0.2, n_samples)
suspicious_flags = np.random.poisson(0.3, n_samples)
 
# Label: 1 = intrusion if many failed logins or suspicious flags with long duration or large transfers
intrusion = ((failed_logins > 1) | (suspicious_flags > 1) | (duration > 5) & (bytes_sent > 2000)).astype(int)
 
# Stack features and labels
X = np.stack([duration, bytes_sent, bytes_received, failed_logins, suspicious_flags], axis=1)
y = intrusion
 
# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
 
# Build classifier model
model = models.Sequential([
    layers.Input(shape=(5,)),
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(1, activation='sigmoid')  # 0 = normal, 1 = intrusion
])
 
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=15, batch_size=32, verbose=0)
 
# Evaluate
loss, acc = model.evaluate(X_test, y_test)
print(f"✅ Intrusion Detection Accuracy: {acc:.4f}")
 
# Predict traffic status
preds = (model.predict(X_test[:5]) > 0.5).astype(int).flatten()
for i in range(5):
    print(f"Traffic {i+1}: {'🚨 Intrusion' if preds[i] else '✅ Normal'} (Actual: {'🚨 Intrusion' if y_test[i] else '✅ Normal'})")