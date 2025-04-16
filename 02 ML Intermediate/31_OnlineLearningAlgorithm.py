"""
Project 71. Online learning algorithm implementation
Description:
An online learning algorithm updates its model incrementally as new data arrives, making it ideal for streaming or real-time systems. In this project, we implement an online logistic regression classifier using SGDClassifier from scikit-learn, which supports learning in mini-batches.

About:
🔁 What This Project Demonstrates:
Implements online/incremental learning with partial_fit

Simulates streamed training in batches

Tracks batch-wise model accuracy and final performance
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
 
# Generate a simulated binary classification dataset
X, y = make_classification(n_samples=1000, n_features=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
 
# Initialize an online learning model with logistic regression loss
model = SGDClassifier(loss='log_loss', learning_rate='optimal', max_iter=1, warm_start=True)
 
# Train the model in small batches to simulate streaming
batch_size = 100
n_batches = int(np.ceil(len(X_train) / batch_size))

print("Training with online batches...\n")
for i in range(n_batches):
    start = i * batch_size
    end = start + batch_size
    X_batch = X_train[start:end]
    y_batch = y_train[start:end]
    model.partial_fit(X_batch, y_batch, classes=np.unique(y))  # Online update
    acc = model.score(X_batch, y_batch)
    print(f"Batch {i+1}/{n_batches} Accuracy: {acc:.2f}")
 
# Evaluate on test set
y_pred = model.predict(X_test)
print("\nFinal Evaluation on Test Data:")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")