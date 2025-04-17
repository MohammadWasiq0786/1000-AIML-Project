"""
Project 768: On-Device Learning Implementation
Description
On-device learning enables models to adapt or personalize directly on edge devices without relying on cloud servers. It’s ideal for privacy-sensitive applications (e.g., personal assistants, wearables). We simulate this with online learning using a lightweight model and incremental updates from streaming input. This is demonstrated using scikit-learn with SGDClassifier to mimic real-time on-device updates.

This simulates on-device adaptation, where the model gets better as more user data becomes available. In actual embedded applications, this might use tiny neural networks with micro-controllers and Edge Impulse or TensorFlow Lite supporting online fine-tuning.
"""

from sklearn.linear_model import SGDClassifier
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import numpy as np
 
# Load a small image dataset (like MNIST but smaller) for simulation
digits = load_digits()
X, y = digits.data, digits.target
 
# Normalize features
scaler = StandardScaler()
X = scaler.fit_transform(X)
 
# Simulate a stream by splitting into chunks
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
stream_chunks = np.array_split(X_train, 10)
label_chunks = np.array_split(y_train, 10)
 
# Initialize a linear model for online learning
model = SGDClassifier(loss='log_loss')  # Logistic regression
 
# Use partial_fit to train incrementally, simulating on-device updates
classes = np.unique(y_train)
for i, (X_chunk, y_chunk) in enumerate(zip(stream_chunks, label_chunks)):
    model.partial_fit(X_chunk, y_chunk, classes=classes)
    acc = accuracy_score(y_test, model.predict(X_test))
    print(f"✅ After update {i+1}, accuracy: {acc:.4f}")
 
# Final accuracy after all updates
final_acc = accuracy_score(y_test, model.predict(X_test))
print(f"\n✅ Final model accuracy after simulated on-device learning: {final_acc:.4f}")