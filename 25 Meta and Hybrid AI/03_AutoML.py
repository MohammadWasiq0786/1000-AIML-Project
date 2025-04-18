"""
Project 963: AutoML Implementation
Description
AutoML (Automated Machine Learning) automates the process of selecting models, hyperparameters, and feature engineering, making machine learning more accessible and efficient. In this project, we'll implement a simple AutoML pipeline that can automatically choose models and hyperparameters for a classification task.

Key Concepts Covered:
AutoML with TPOT: Automatically tunes models and selects the best-performing pipeline using evolutionary algorithms.

Genetic Algorithm: A search method that uses the concept of survival of the fittest to evolve models.

Model Export: TPOT allows you to export the best pipeline to a Python script, which can be reused for predictions.
"""

import numpy as np
from tpot import TPOTClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
 
# Load the digits dataset (for simplicity)
digits = load_digits()
X, y = digits.data, digits.target
 
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
 
# Preprocess the data (standardization)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
 
# Initialize the AutoML model (TPOTClassifier)
model = TPOTClassifier( generations=5, population_size=20, random_state=42, verbosity=2)
 
# Fit the model to the training data
model.fit(X_train, y_train)
 
# Evaluate the model on the test data
accuracy = model.score(X_test, y_test)
print(f"✅ AutoML Model Accuracy: {accuracy:.4f}")
 
# Export the best pipeline
model.export('best_model_pipeline.py')