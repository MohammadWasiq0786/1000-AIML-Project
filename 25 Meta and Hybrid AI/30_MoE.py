"""
Project 990: Mixture of Experts Implementation
Description
A Mixture of Experts (MoE) model is a type of ensemble model where different "expert" models are trained on different parts of the input space. The model uses a gating mechanism to decide which expert should make the prediction for each input. This allows the model to allocate resources to different tasks efficiently and improve performance on a variety of problems. In this project, we will implement a Mixture of Experts model using a gating network to combine the predictions of multiple expert models.

Key Concepts Covered:
Mixture of Experts (MoE): An ensemble learning technique where multiple expert models specialize in different parts of the input space. A gating mechanism decides which expert is best suited for each input.

Gating Network: A model (often simple) that learns to assign inputs to the correct expert based on features.

Ensemble Learning: MoE is a type of ensemble method where the "experts" collaborate based on the gating network's decision, leading to better predictions than individual models.
"""

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
 
# Load the dataset
data = load_breast_cancer()
X = data.data
y = data.target
 
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
 
# Define expert models
experts = [
    ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
    ('gb', GradientBoostingClassifier(n_estimators=100, random_state=42)),
]
 
# Train each expert model
for name, model in experts:
    model.fit(X_train, y_train)
 
# Define the gating model (simple logistic regression for simplicity)
gating_model = LogisticRegression()
gating_model.fit(X_train, y_train)
 
# Get the predictions from each expert
expert_predictions = np.zeros((len(X_test), len(experts)))
for i, (name, model) in enumerate(experts):
    expert_predictions[:, i] = model.predict_proba(X_test)[:, 1]  # Probability for class 1
 
# Use the gating model to predict which expert's prediction to trust for each instance
gating_preds = gating_model.predict_proba(X_test)[:, 1]
 
# Combine expert predictions using gating model's output
final_predictions = np.zeros(len(X_test))
for i in range(len(X_test)):
    expert_idx = np.argmax(expert_predictions[i, :])  # Choose the expert with highest confidence
    final_predictions[i] = expert_predictions[i, expert_idx]  # Trust the selected expert's prediction
 
# Evaluate the Mixture of Experts model
accuracy = accuracy_score(y_test, final_predictions.round())  # Convert probabilities to binary predictions
print(f"Accuracy of Mixture of Experts model: {accuracy:.4f}")