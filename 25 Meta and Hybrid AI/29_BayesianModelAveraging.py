"""
Project 989: Bayesian Model Averaging
Description
Bayesian Model Averaging (BMA) is a method for combining multiple models based on their posterior probabilities. Instead of picking a single "best" model, BMA takes the weighted average of the predictions of all candidate models, where the weight for each model is proportional to its posterior probability. This approach improves predictive performance by accounting for model uncertainty.

In this project, we will implement Bayesian Model Averaging using a simple example with different models and weight their predictions based on their posterior probabilities.

Python Implementation with Comments (Bayesian Model Averaging)
We'll use scikit-learn models and compute the posterior probability of each model using its performance (e.g., cross-validation accuracy). We'll then combine their predictions using these weights.

Key Concepts Covered:
Bayesian Model Averaging (BMA): A method that combines the predictions of multiple models weighted by their posterior probabilities. It accounts for uncertainty in model selection.

Posterior Probability: In BMA, the posterior probability is often estimated based on model performance (e.g., cross-validation accuracy).

Weighted Averaging: The predictions of different models are weighted by their posterior probabilities, and these predictions are combined for final output.
"""

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.base import clone
 
# Load the dataset
data = load_breast_cancer()
X = data.data
y = data.target
 
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
 
# Define base models for Bayesian Model Averaging
models = [
    ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
    ('gb', GradientBoostingClassifier(n_estimators=100, random_state=42)),
    ('lr', LogisticRegression(max_iter=1000, random_state=42))
]
 
# Step 1: Compute model performance (posterior probability) via cross-validation
model_scores = {}
for name, model in models:
    score = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    model_scores[name] = np.mean(score)
 
# Step 2: Compute posterior probabilities (normalize the scores)
total_score = sum(model_scores.values())
model_weights = {name: score / total_score for name, score in model_scores.items()}
 
# Step 3: Train each model on the full training set
trained_models = {}
for name, model in models:
    trained_models[name] = clone(model).fit(X_train, y_train)
 
# Step 4: Make predictions using each model and combine predictions using weights
def weighted_prediction(models, weights, X):
    # Weighted average of predictions
    weighted_preds = np.zeros((X.shape[0], len(models)))
    for i, (name, model) in enumerate(models):
        pred = trained_models[name].predict_proba(X)[:, 1]  # Get probability for class 1
        weighted_preds[:, i] = pred * weights[name]
    
    # Combine predictions (sum of weighted predictions)
    final_prediction = np.sum(weighted_preds, axis=1)
    return (final_prediction > 0.5).astype(int)  # Convert to binary class prediction
 
# Step 5: Make predictions on the test set
y_pred = weighted_prediction(models, model_weights, X_test)
 
# Step 6: Evaluate the accuracy of the Bayesian Model Averaging
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy of Bayesian Model Averaging: {accuracy:.4f}")