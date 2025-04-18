"""
Project 997: Online Learning Implementation
Description
Online learning is a machine learning paradigm where the model is trained incrementally on data as it arrives. This is particularly useful for applications where data is continuously generated or when the dataset is too large to fit into memory. In this project, we will implement an online learning system using a simple model that is updated incrementally as new data arrives, instead of training on the entire dataset at once.

Key Concepts Covered:
Online Learning: In online learning, the model is updated incrementally as new data points arrive, without needing to retrain on the entire dataset.

Stochastic Gradient Descent (SGD): A type of online learning where the model's parameters are updated after each mini-batch or individual data point.

Incremental Model Update: The model is updated using the partial_fit method, which allows learning from data in batches or one point at a time.
"""

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
import random
 
# Load the Iris dataset
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = iris.target
 
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
 
# Initialize the SGDClassifier for online learning
online_model = SGDClassifier(loss='log', max_iter=1000, random_state=42)
 
# Train the model incrementally using mini-batches (simulating the arrival of data in chunks)
for i in range(0, len(X_train), 10):  # Using mini-batches of size 10
    batch_X = X_train.iloc[i:i+10]
    batch_y = y_train.iloc[i:i+10]
    
    # Update the model with the new mini-batch of data
    online_model.partial_fit(batch_X, batch_y, classes=np.unique(y))
 
# Evaluate the model on the test set
y_pred = online_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy after Online Learning: {accuracy:.4f}")
 
# Simulate new incoming data (e.g., after model deployment)
new_data_X = pd.DataFrame([[5.7, 3.0, 4.2, 1.2]])  # New data point
new_data_y = [1]  # True label for the new data point
 
# Update the model incrementally with the new data
online_model.partial_fit(new_data_X, new_data_y)
 
# Evaluate the model again after learning from the new data
y_pred_new = online_model.predict(X_test)
accuracy_new = accuracy_score(y_test, y_pred_new)
print(f"Accuracy after learning from new data: {accuracy_new:.4f}")