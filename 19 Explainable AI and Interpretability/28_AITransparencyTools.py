"""
Project 748: AI Transparency Tools
Description:
AI transparency refers to the ability to understand and explain how a machine learning model makes decisions. As AI systems are increasingly deployed in high-stakes domains such as healthcare, finance, and law enforcement, it's essential that these systems are transparent, so users and stakeholders can trust and verify their decisions. In this project, we will implement AI transparency tools to analyze and explain a model's behavior, using techniques such as model introspection, visualization of decision-making processes, and decision rules extraction. We will build a transparency toolkit that includes tools for visualizing model behavior and explaining its predictions in a comprehensible way.


Explanation:
Dataset Loading and Preprocessing: We load the Iris dataset, which contains 150 samples of iris flowers across three species. The dataset is then split into training and testing sets.

Model Training: A Random Forest classifier is trained on the dataset, with 100 trees in the forest.

Partial Dependence Plots (PDP): The visualize_partial_dependence() function uses plot_partial_dependence from Scikit-learn to generate PDPs. PDPs show the relationship between a feature and the predicted outcome, helping us understand how individual features impact the model's predictions. For instance, we can visualize how the petal length affects the model's decision to classify the flower species.

SHAP for Model Explainability: The explain_with_shap() function uses SHAP (SHapley Additive exPlanations) to explain the predictions of the Random Forest model. The TreeExplainer is used to calculate SHAP values, which indicate how much each feature contributes to a specific prediction. The SHAP summary plot visualizes the contribution of each feature to the model's decision.

Model Transparency: Both PDP and SHAP visualization techniques allow us to introspect the model's decision-making process, improving transparency. PDP helps visualize the effect of individual features, while SHAP provides detailed feature contributions for each prediction.

These tools help ensure that the Random Forest model is interpretable, making it easier for stakeholders to understand how the model reaches its decisions.
"""

import numpy as np
import matplotlib.pyplot as plt
import shap
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import plot_partial_dependence
 
# 1. Load and preprocess the dataset (Iris dataset for simplicity)
def load_dataset():
    data = load_iris()
    X = data.data
    y = data.target
    return X, y, data.feature_names
 
# 2. Train a Random Forest classifier
def train_model(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model
 
# 3. Visualize Partial Dependence (PDP) for model transparency
def visualize_partial_dependence(model, X_train, feature_names):
    """
    Visualize Partial Dependence Plots (PDP) to explain how individual features influence the model's predictions.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    plot_partial_dependence(model, X_train, features=[0, 1, 2, 3], feature_names=feature_names, ax=ax)
    plt.suptitle('Partial Dependence Plots (PDP)')
    plt.show()
 
# 4. Explain model predictions using SHAP (Tree Explainer for Random Forest)
def explain_with_shap(model, X_train):
    """
    Use SHAP to explain the predictions of a Random Forest model.
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train)
    
    # Visualize SHAP summary plot for transparency
    shap.summary_plot(shap_values, X_train, feature_names=["sepal length", "sepal width", "petal length", "petal width"])
 
# 5. Example usage
X, y, feature_names = load_dataset()
 
# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
 
# Train the Random Forest model
model = train_model(X_train, y_train)
 
# Visualize the Partial Dependence Plots (PDP)
visualize_partial_dependence(model, X_train, feature_names)
 
# Explain model predictions using SHAP
explain_with_shap(model, X_train)