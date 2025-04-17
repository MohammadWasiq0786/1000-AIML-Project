"""
Project 731: Model-Agnostic Interpretability Methods
Description:
Model-agnostic interpretability methods are techniques that can be applied to any machine learning model, regardless of its internal workings. These methods aim to provide explanations for the model's predictions in a human-understandable way. Common approaches include LIME, SHAP, and Partial Dependence Plots (PDP). In this project, we will explore a model-agnostic interpretability method to understand and explain the predictions of a black-box model (e.g., Random Forest, SVM, Neural Networks) using SHAP (SHapley Additive exPlanations), which is one of the most widely used model-agnostic methods.

Explanation:
Data Preprocessing: We load and preprocess the Iris dataset, which consists of 4 features and 3 target classes. This dataset is used to train the machine learning model.

Model Training: We train a Random Forest classifier on the Iris dataset. Random Forest is a powerful ensemble model for classification tasks.

SHAP Values Calculation: We use SHAP (SHapley Additive exPlanations) to calculate SHAP values. SHAP provides both local and global explanations for model predictions. Local explanations explain why the model made a specific prediction for an individual instance, while global explanations provide insights into which features are most important across all predictions.

Visualization:

The summary plot visualizes the distribution of SHAP values for each feature across all instances in the dataset. This plot helps identify which features contribute the most to the model's predictions.

The bar plot provides a simpler view of feature importance, with the features sorted by their average SHAP values.

By using SHAP, we gain a clear understanding of how different features contribute to the predictions made by a complex model like Random Forest, making the model more interpretable and easier to explain to stakeholders.
"""

import shap
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
 
# 1. Load and preprocess the dataset (Iris dataset for simplicity)
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
 
# 3. Use SHAP to explain the model's predictions
def explain_model_with_shap(model, X_train):
    """
    Use SHAP to explain the predictions of a model.
    This method provides global and local explanations for the model.
    """
    explainer = shap.TreeExplainer(model)  # SHAP explainer for tree-based models like RandomForest
    shap_values = explainer.shap_values(X_train)
    return shap_values
 
# 4. Visualize SHAP values for the feature importance and individual predictions
def visualize_shap_values(shap_values, feature_names):
    """
    Visualize SHAP values using SHAP summary plots and bar plots.
    """
    # Summary plot of SHAP values (global feature importance)
    shap.summary_plot(shap_values[1], feature_names=feature_names)  # Class 1 (Versicolor) for explanation
    
    # SHAP bar plot for feature importance
    shap.summary_bar_plot(shap_values[1], feature_names=feature_names)
 
# 5. Example usage
X, y, feature_names = load_dataset()
 
# Split dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
 
# Train the Random Forest model
model = train_model(X_train, y_train)
 
# Explain the model's predictions with SHAP
shap_values = explain_model_with_shap(model, X_train)
 
# Visualize the SHAP values for feature importance and individual predictions
visualize_shap_values(shap_values, feature_names)