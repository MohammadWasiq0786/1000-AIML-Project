"""
Project 988: Stacking Ensemble Implementation
Description
Stacking is an ensemble learning technique where multiple base models (diverse algorithms) are trained, and their predictions are combined using a meta-model (often a simple model like logistic regression or another machine learning algorithm). The goal is to leverage the strengths of different models to improve predictive performance.

In this project, we will implement a stacking ensemble using base models like Random Forest and Gradient Boosting, and combine them using a Logistic Regression meta-model.

Key Concepts Covered:
Stacking: Stacking combines the predictions of multiple base models by training a meta-model (often called a blender or final estimator) to learn the best way to combine those predictions. It’s one of the most powerful ensemble learning methods.

Base Models: These are the individual models (e.g., Random Forest and Gradient Boosting) that make predictions.

Meta-model: This is the model that combines the predictions from the base models. In our case, we used Logistic Regression.

Diversity of Models: By combining different algorithms, stacking can leverage the strengths of each model, often leading to improved predictive accuracy.
"""

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
 
# Load the breast cancer dataset
data = load_breast_cancer()
X = data.data
y = data.target
 
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
 
# Define the base learners (Random Forest and Gradient Boosting)
base_learners = [
    ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
    ('gb', GradientBoostingClassifier(n_estimators=100, random_state=42))
]
 
# Define the meta-learner (Logistic Regression)
meta_learner = LogisticRegression()
 
# Create the stacking ensemble model
stacking_model = StackingClassifier(estimators=base_learners, final_estimator=meta_learner)
 
# Train the stacking model
stacking_model.fit(X_train, y_train)
 
# Make predictions with the stacking model
stacking_pred = stacking_model.predict(X_test)
 
# Evaluate the performance of the stacking model
stacking_acc = accuracy_score(y_test, stacking_pred)
print(f"Stacking Ensemble Accuracy: {stacking_acc:.4f}")