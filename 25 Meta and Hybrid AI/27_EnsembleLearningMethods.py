"""
Project 987: Ensemble Learning Methods
Description
Ensemble learning combines multiple models to improve performance, stability, and generalization. The core idea is to leverage the diversity of multiple models, reducing overfitting and bias. In this project, we will implement common ensemble learning methods such as Bagging, Boosting, and Stacking. We’ll use Random Forests (for Bagging) and Gradient Boosting (for Boosting) as examples.

Bagging (Bootstrap Aggregating): Using Random Forests to average predictions from multiple decision trees.

Boosting: Using Gradient Boosting to combine weak models sequentially, improving performance.

Stacking: Combining multiple models (e.g., Random Forest and Gradient Boosting) using a meta-learner.

Key Concepts Covered:
Bagging (Bootstrap Aggregating): This method creates multiple models from random subsets of the training data and averages their predictions. It reduces variance and helps with overfitting. Random Forests are a popular example of bagging.

Boosting: Boosting combines weak learners sequentially, with each model focusing on the errors of the previous model. Gradient Boosting is a widely used boosting method.

Stacking: Stacking involves combining different models (base learners) by training a meta-model on their predictions. This often improves performance by utilizing the strengths of multiple models.
"""


import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
 
# Load dataset
data = load_breast_cancer()
X = data.data
y = data.target
 
# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
 
# 1. Bagging: Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
 
# 2. Boosting: Gradient Boosting Classifier
gb_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
 
# 3. Stacking: Using Random Forest and Gradient Boosting as base models, with Logistic Regression as meta-model
base_learners = [
    ('rf', RandomForestClassifier(n_estimators=50, random_state=42)),
    ('gb', GradientBoostingClassifier(n_estimators=50, random_state=42))
]
stacking_model = StackingClassifier(estimators=base_learners, final_estimator=LogisticRegression())
 
# Train models
rf_model.fit(X_train, y_train)
gb_model.fit(X_train, y_train)
stacking_model.fit(X_train, y_train)
 
# Make predictions
rf_pred = rf_model.predict(X_test)
gb_pred = gb_model.predict(X_test)
stacking_pred = stacking_model.predict(X_test)
 
# Evaluate models
rf_acc = accuracy_score(y_test, rf_pred)
gb_acc = accuracy_score(y_test, gb_pred)
stacking_acc = accuracy_score(y_test, stacking_pred)
 
print(f"Random Forest Accuracy: {rf_acc:.4f}")
print(f"Gradient Boosting Accuracy: {gb_acc:.4f}")
print(f"Stacking Model Accuracy: {stacking_acc:.4f}")