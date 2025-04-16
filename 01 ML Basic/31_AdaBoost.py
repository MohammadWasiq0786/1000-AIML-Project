"""
Project 31. AdaBoost algorithm implementation
Description:
AdaBoost (Adaptive Boosting) is a boosting ensemble technique that combines multiple weak learners (typically decision stumps) to create a strong classifier. Each learner is trained focusing more on the samples that were misclassified by previous models. In this project, we implement AdaBoost using scikit-learn with a base Decision Tree on the Iris dataset.

About:
⚙️ What This Project Demonstrates:
Builds multiple weak learners (decision stumps)

Adjusts focus toward misclassified samples at each round

Produces a strong classifier through adaptive boosting
"""

# Import necessary libraries
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
 
# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target
 
# Use only two classes for binary classification (Setosa vs Versicolor)
X_binary = X[y != 2]
y_binary = y[y != 2]
 
# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_binary, y_binary, test_size=0.3, random_state=42)
 
# Create base estimator (shallow decision tree)
base_estimator = DecisionTreeClassifier(max_depth=1)
 
# Initialize AdaBoost with base estimator
ada_model = AdaBoostClassifier(
    estimator=base_estimator,
    n_estimators=50,
    learning_rate=1.0,
    random_state=42
)
 
# Train the model
ada_model.fit(X_train, y_train)
 
# Make predictions
y_pred = ada_model.predict(X_test)
 
# Evaluate the model
print(f"AdaBoost Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=iris.target_names[:2]))