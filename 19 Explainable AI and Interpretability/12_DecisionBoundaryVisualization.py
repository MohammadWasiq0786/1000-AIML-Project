"""
Project 732: Decision Boundary Visualization
Description:
Decision boundary visualization is a technique used to visualize the regions of feature space where a machine learning model makes different predictions. It helps in understanding how the model separates different classes based on the input features. This is particularly useful for classification models to see how well they generalize across different regions of the feature space. In this project, we will visualize the decision boundary of a Random Forest classifier trained on a 2D dataset like the Iris dataset (using only two features for simplicity).

Explanation:
Data Preprocessing: We load the Iris dataset but only use the first two features (sepal length and sepal width) for simplicity, as this makes it easier to visualize the decision boundary in 2D.

Model Training: We train a Random Forest classifier on the selected features of the Iris dataset.

Decision Boundary Visualization:

We create a mesh grid that covers the entire feature space (from the minimum to the maximum values of the features).

We then predict the class label for each point in the grid using the trained Random Forest model.

The contour plot represents the decision boundary where the model changes its predictions from one class to another.

The scatter plot visualizes the actual data points, showing how they are classified by the model in relation to the decision boundary.

This technique helps us understand how the model divides the feature space and makes predictions, providing a visual interpretation of the model’s decision-making process.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
 
# 1. Load and preprocess the Iris dataset (using only two features for simplicity)
def load_dataset():
    data = load_iris()
    X = data.data[:, :2]  # Use only the first two features (sepal length and sepal width)
    y = data.target
    feature_names = data.feature_names[:2]  # Names of the two selected features
    return X, y, feature_names
 
# 2. Train a Random Forest classifier
def train_model(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model
 
# 3. Visualize the decision boundary
def plot_decision_boundary(model, X, y, feature_names):
    """
    Visualizes the decision boundary of a classifier on a 2D feature space.
    """
    # Create a mesh grid to plot the decision boundary
    h = .02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
 
    # Predict the class for each point on the mesh grid
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
 
    # Plot the decision boundary
    plt.contourf(xx, yy, Z, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', marker='o', s=100)
    plt.title("Decision Boundary Visualization")
    plt.xlabel(feature_names[0])
    plt.ylabel(feature_names[1])
    plt.show()
 
# 4. Example usage
X, y, feature_names = load_dataset()
 
# Split dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
 
# Train the Random Forest model
model = train_model(X_train, y_train)
 
# Visualize the decision boundary
plot_decision_boundary(model, X_train, y_train, feature_names)