"""
Project 721: Feature Importance Visualization
Description:
Feature importance visualization helps to identify and understand which features (input variables) in a model contribute the most to its predictions. This is especially useful in machine learning models, where certain features might play a crucial role in driving the decision-making process. In this project, we will implement a method to visualize feature importance, which can help in understanding and interpreting the results of machine learning models, especially black-box models like Random Forests and Gradient Boosting.

In this implementation, we will use a Random Forest classifier to perform classification on a dataset (e.g., Iris Dataset) and visualize the importance of features using the feature_importances_ attribute provided by scikit-learn.

Explanation:
Load the Dataset: We use the Iris dataset as an example. It contains 4 features (sepal length, sepal width, petal length, and petal width) and is commonly used for classification tasks.

Train the Model: We train a Random Forest classifier on the dataset. The RandomForestClassifier is a popular model that provides feature importance through its feature_importances_ attribute.

Visualize Feature Importance: We use matplotlib to create a bar chart showing the importance of each feature. The features are sorted by their importance, and the chart is displayed with relative importance values.

This method works well for tree-based models like Random Forests and Gradient Boosting, where the feature_importances_ attribute can be directly accessed. For other models, different techniques, such as permutation importance, may be used.
"""



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
 
# 3. Visualize feature importance
def visualize_feature_importance(model, feature_names):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
 
    plt.figure(figsize=(10, 6))
    plt.title("Feature Importance Visualization")
    plt.barh(range(len(indices)), importances[indices], align="center")
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel("Relative Importance")
    plt.show()
 
# 4. Example usage
X, y, feature_names = load_dataset()
 
# Split dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
 
# Train the Random Forest model
model = train_model(X_train, y_train)
 
# Visualize the feature importance
visualize_feature_importance(model, feature_names)