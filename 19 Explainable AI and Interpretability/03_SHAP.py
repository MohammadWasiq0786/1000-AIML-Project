"""
Project 723: SHAP Values for Model Interpretation
Description:
SHAP (SHapley Additive exPlanations) values provide a unified measure of feature importance, derived from game theory. SHAP values help in understanding how much each feature contributes to a specific prediction made by a model. Unlike other feature importance methods, SHAP values can explain individual predictions in a consistent manner. In this project, we will implement SHAP to interpret a Random Forest model's predictions on a dataset like the Iris dataset.

Explanation:
Load the Dataset: We use the Iris dataset, which is a simple classification dataset where the goal is to classify flower species based on various features.

Train the Model: We train a Random Forest classifier on the Iris dataset.

Compute SHAP Values: The SHAP TreeExplainer is used to compute the SHAP values. It works well for tree-based models like Random Forest and XGBoost. The SHAP values tell us the contribution of each feature to the model's output for each prediction.

Visualize SHAP Values: The summary plot from SHAP visualizes the feature importance across all predictions. The plot shows how much each feature impacts the predictions and helps in understanding the relative importance of each feature.

SHAP provides both global feature importance (as shown in the summary plot) and local explanations for individual predictions, making it a powerful tool for explaining complex models.
"""

import shap
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
 
# 1. Load the dataset (Iris dataset for simplicity)
def load_dataset():
    data = load_iris()
    X = data.data
    y = data.target
    feature_names = data.feature_names
    return X, y, feature_names
 
# 2. Train a Random Forest classifier
def train_model(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model
 
# 3. Compute SHAP values using the SHAP library
def compute_shap_values(model, X_train):
    # Create a SHAP explainer for the Random Forest model
    explainer = shap.TreeExplainer(model)
    
    # Compute SHAP values for the training data
    shap_values = explainer.shap_values(X_train)
    return shap_values
 
# 4. Visualize SHAP values for a single instance
def visualize_shap_values(shap_values, feature_names):
    # Summary plot for feature importance
    shap.summary_plot(shap_values[1], feature_names=feature_names)  # Class 1 (Versicolor) for explanation
 
# 5. Example usage
X, y, feature_names = load_dataset()
 
# Split dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
 
# Train the Random Forest model
model = train_model(X_train, y_train)
 
# Compute SHAP values
shap_values = compute_shap_values(model, X_train)
 
# Visualize SHAP values (feature importance)
visualize_shap_values(shap_values, feature_names)