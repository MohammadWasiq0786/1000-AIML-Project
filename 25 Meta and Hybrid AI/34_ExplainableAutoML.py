"""
Project 994: Explainable AutoML
Description
Explainable AI (XAI) seeks to make machine learning models more transparent and interpretable. In Explainable AutoML, we integrate explainability methods into the AutoML pipeline to provide insight into how models make predictions. In this project, we will use TPOT (AutoML) along with explainability techniques like SHAP (SHapley Additive exPlanations) to understand the decision-making process of the automatically selected model.

Key Concepts Covered:
AutoML with Explainability: Integrating explainability into automated machine learning workflows to make the results interpretable and transparent.

SHAP: SHAP provides a game-theoretic approach to explain the output of machine learning models, helping to understand how each feature contributes to the predictions.

Interpretability: In this case, the focus is on interpreting the predictions of the best model selected by AutoML and providing visual insights into model decision-making.
"""

# pip install tpot shap

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from tpot import TPOTClassifier
import shap
import matplotlib.pyplot as plt
 
# Load the Iris dataset
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = iris.target
 
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
 
# Initialize the TPOTClassifier with a generation of 5 and population size of 20
tpot = TPOTClassifier(generations=5, population_size=20, random_state=42, verbosity=2)
 
# Fit the TPOTClassifier to the training data (automated model selection)
tpot.fit(X_train, y_train)
 
# Evaluate the model on the test data
y_pred = tpot.predict(X_test)
 
# Accuracy of the best model found by TPOT
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy of the best model found by TPOT: {accuracy:.4f}")
 
# Export the best model pipeline found by TPOT
tpot.export('best_model_pipeline.py')
 
# Step 2: Use SHAP to explain the predictions of the best model
# Create the explainer object for the model (using the best model found by TPOT)
explainer = shap.KernelExplainer(tpot.fitted_pipeline_.predict_proba, X_train)
 
# Explain the predictions for the test set
shap_values = explainer.shap_values(X_test)
 
# Plot the SHAP summary plot to understand feature importance
shap.summary_plot(shap_values, X_test)
 
# Plot the SHAP values for the first instance in the test set (local explanation)
shap.initjs()  # Initialize JavaScript for SHAP plots
shap.force_plot(explainer.expected_value[0], shap_values[0][0], X_test.iloc[0,:])