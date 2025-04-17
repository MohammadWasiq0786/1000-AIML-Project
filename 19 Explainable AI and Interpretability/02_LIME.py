"""
Project 722: LIME Implementation for Model Explanation
Description:
LIME (Local Interpretable Model-agnostic Explanations) is a method that explains the predictions of machine learning models by approximating the model locally with an interpretable model, such as a linear regression or decision tree. LIME works by perturbing the input data, observing the changes in predictions, and then training a surrogate model that approximates the original model’s behavior in that local region. This is especially useful for black-box models like Random Forests, SVMs, or deep learning models, where direct interpretability is difficult. In this project, we will implement LIME to explain a Random Forest model's predictions on a sample dataset.

Explanation:
Load the Dataset: We load the Iris dataset, which is a simple classification task where the goal is to predict the species of a flower based on its features (sepal length, sepal width, petal length, and petal width).

Train the Model: We train a Random Forest classifier on the Iris dataset.

Create a LIME Explainer: The LIME explainer is created using the LimeTabularExplainer. It takes the training data and labels, feature names, class names, and other parameters to explain the model's behavior locally.

Explain an Instance: We select an instance from the test set, and use the LIME explainer to generate and visualize an explanation for the model's prediction. The explanation includes feature importances for the instance, showing which features had the most influence on the prediction.

The LIME explainer shows how a locally interpretable model (a surrogate model) is used to approximate the Random Forest model's behavior for a specific instance. This is particularly helpful for explaining individual predictions from complex models.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from lime.lime_tabular import LimeTabularExplainer
 
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
 
# 3. Create a LIME explainer
def explain_prediction(model, X_train, feature_names):
    explainer = LimeTabularExplainer(X_train, 
                                     training_labels=None, 
                                     feature_names=feature_names, 
                                     class_names=["Setosa", "Versicolor", "Virginica"], 
                                     discretize_continuous=True)
    
    return explainer
 
# 4. Get and visualize explanation for a single instance
def explain_instance(model, explainer, instance, instance_idx=0):
    explanation = explainer.explain_instance(instance, model.predict_proba, num_features=4)
    
    # Visualize the explanation
    explanation.show_in_notebook(show_table=True, show_all=False)
 
# 5. Example usage
X, y, feature_names = load_dataset()
 
# Split dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
 
# Train the Random Forest model
model = train_model(X_train, y_train)
 
# Create a LIME explainer
explainer = explain_prediction(model, X_train, feature_names)
 
# Choose an instance to explain
instance = X_test[0]  # Use the first test instance
 
# Explain the chosen instance
explain_instance(model, explainer, instance)