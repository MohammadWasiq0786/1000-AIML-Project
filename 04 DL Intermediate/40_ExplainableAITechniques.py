"""
Project 160. Explainable AI techniques
Description:
Explainable AI (XAI) helps humans understand how and why machine learning models make predictions. This project demonstrates core techniques like LIME (Local Interpretable Model-Agnostic Explanations) and SHAP (SHapley Additive exPlanations) to explain predictions of a trained classifier on tabular data (e.g., breast cancer dataset).

About:
🧠 What This Project Demonstrates:
Uses LIME to provide local explanation by approximating model locally

Uses SHAP to show feature importance using Shapley values from game theory

Visualizes summary plots, force plots, and individual decisions
"""

# Install if not already: pip install scikit-learn lime shap matplotlib
 
import numpy as np
import pandas as pd
import shap
import lime
import lime.lime_tabular
import matplotlib.pyplot as plt
 
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
 
# Load dataset
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target
feature_names = data.feature_names
 
# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
 
# Train a black-box model
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)
 
# Pick an instance to explain
idx = 10
instance = X_test.iloc[idx:idx+1].values
 
# -----------------------
# 🔍 LIME Explanation
# -----------------------
explainer_lime = lime.lime_tabular.LimeTabularExplainer(
    training_data=np.array(X_train),
    feature_names=feature_names,
    class_names=data.target_names,
    mode="classification"
)
 
exp = explainer_lime.explain_instance(instance[0], model.predict_proba, num_features=5)
print("🧠 LIME Explanation:")
exp.show_in_notebook(show_table=True)
 
# -----------------------
# 🔍 SHAP Explanation
# -----------------------
explainer_shap = shap.TreeExplainer(model)
shap_values = explainer_shap.shap_values(X_test)
 
print("🧠 SHAP Explanation (summary plot):")
shap.summary_plot(shap_values[1], X_test, feature_names=feature_names)
 
# Explanation for a single instance
print(f"🧠 SHAP Explanation for sample {idx}:")
shap.force_plot(explainer_shap.expected_value[1], shap_values[1][idx], X_test.iloc[idx], matplotlib=True)