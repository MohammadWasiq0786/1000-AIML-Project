"""
Project 730: Decision Tree Extraction from Black-Box Models
Description:
Decision tree extraction from black-box models is the process of approximating the behavior of a complex model (e.g., deep neural networks, random forests) by using a simpler, interpretable model like a decision tree. This technique is particularly useful when we want to explain the decisions made by a complex model in terms of easy-to-understand rules, which is important for model interpretability and trustworthiness. In this project, we will train a Random Forest model and extract an approximating decision tree to explain its decision-making process.

Explanation:
Data Preprocessing: We load and preprocess the Iris dataset, which consists of 4 features (sepal length, sepal width, petal length, petal width) and 3 classes (flower species).

Train the Random Forest Model: We train a Random Forest classifier on the Iris dataset. The Random Forest model consists of multiple decision trees that can be used to approximate more complex decision boundaries.

Extract Decision Tree: We extract one of the individual decision trees from the Random Forest using the estimators_ attribute. This decision tree can be used as an approximation of the Random Forest model’s behavior.

Visualization: The plot_tree() function is used to visualize the decision tree, where the feature names and class labels are shown. The tree structure represents the learned decision rules for classification.

This technique is useful for interpreting ensemble models like Random Forests by approximating their behavior with a simpler, interpretable decision tree.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
 
# 1. Load and preprocess the dataset (Iris dataset for simplicity)
def load_dataset():
    data = load_iris()
    X = data.data
    y = data.target
    feature_names = data.feature_names
    return X, y, feature_names
 
# 2. Train a Random Forest classifier
def train_random_forest(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model
 
# 3. Extract a decision tree from the random forest
def extract_decision_tree(random_forest_model, X_train, y_train):
    """
    Extract a single decision tree from the Random Forest model.
    The idea is to use the first decision tree in the forest as an approximation.
    """
    tree_model = random_forest_model.estimators_[0]  # Get the first decision tree from the Random Forest
    return tree_model
 
# 4. Visualize the decision tree
def visualize_decision_tree(tree_model, feature_names):
    """
    Visualize the extracted decision tree from the Random Forest model.
    """
    plt.figure(figsize=(12, 8))
    plot_tree(tree_model, feature_names=feature_names, filled=True, rounded=True, class_names=["Setosa", "Versicolor", "Virginica"])
    plt.title("Decision Tree Extracted from Random Forest")
    plt.show()
 
# 5. Example usage
X, y, feature_names = load_dataset()
 
# Split dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
 
# Train the Random Forest model
random_forest_model = train_random_forest(X_train, y_train)
 
# Extract a decision tree from the Random Forest model
decision_tree_model = extract_decision_tree(random_forest_model, X_train, y_train)
 
# Visualize the extracted decision tree
visualize_decision_tree(decision_tree_model, feature_names)