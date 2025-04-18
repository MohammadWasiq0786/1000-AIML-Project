"""
Project 998: Incremental Learning System
Description
Incremental learning is a machine learning approach where the model is trained progressively as new data arrives. This is especially useful in dynamic environments where the data distribution may change over time. In this project, we will implement an incremental learning system where the model continuously learns from incoming batches of data and adapts to changes in the data distribution.

Key Concepts Covered:
Incremental Learning: The model is updated progressively with new data. It allows for continuous adaptation to new patterns without retraining from scratch.

Partial Fit: The partial_fit method is used to update the model incrementally by learning from new data points or mini-batches.

Online Learning: Incremental learning is often used in online learning scenarios, where the data arrives in streams and cannot all be stored at once.
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
 
# Initialize the SGDClassifier for incremental learning
incremental_model = SGDClassifier(loss='log', max_iter=1000, random_state=42)
 
# Simulate incremental learning by updating the model with mini-batches
# We will process the data in batches of 15 samples
for i in range(0, len(X_train), 15):
    batch_X = X_train.iloc[i:i+15]
    batch_y = y_train.iloc[i:i+15]
    
    # Incrementally update the model with each batch of data
    incremental_model.partial_fit(batch_X, batch_y, classes=np.unique(y))
 
# Evaluate the model on the test set
y_pred = incremental_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy after Incremental Learning: {accuracy:.4f}")
 
# Simulate incoming data (e.g., new instances for the model to learn)
new_data_X = pd.DataFrame([[5.8, 2.6, 4.0, 1.2]])  # New data point
new_data_y = [1]  # True label for the new data point
 
# Incrementally update the model with the new data
incremental_model.partial_fit(new_data_X, new_data_y)
 
# Evaluate the model after learning from the new data
y_pred_new = incremental_model.predict(X_test)
accuracy_new = accuracy_score(y_test, y_pred_new)
print(f"Accuracy after learning from new data: {accuracy_new:.4f}")