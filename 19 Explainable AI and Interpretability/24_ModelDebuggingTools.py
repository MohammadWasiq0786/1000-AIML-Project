"""
Project 744: Model Debugging Tools
Description:
Model debugging tools are crucial for understanding, diagnosing, and improving machine learning models. These tools help identify why a model is underperforming, detect issues like overfitting or data leakage, and assist in analyzing the relationships between input features and model predictions. In this project, we will implement model debugging using TensorFlow and Scikit-learn, with tools like learning curves, validation curves, and residual plots to identify potential problems and improve model performance.

Explanation:
Dataset Loading and Preprocessing: We load the Iris dataset and split it into training and test sets using train_test_split. The dataset is ready for training a classification model.

Model Training: We use a Random Forest classifier to train the model on the training set.

Learning Curve: The plot_learning_curve() function generates learning curves to diagnose underfitting or overfitting:

Underfitting is detected when both the training and test accuracy are low.

Overfitting occurs when the training accuracy is much higher than the test accuracy.

A good model will have both training and test accuracy increasing with more data, with the test accuracy stabilizing at a high value.

Validation Curve: The plot_validation_curve() function plots the validation curve for different values of the n_estimators hyperparameter (number of trees in the forest). This helps assess the model's complexity and shows how increasing the number of estimators affects performance.

Performance Evaluation: We evaluate the model's accuracy on the test set to assess its overall performance.

These debugging tools allow us to assess whether the model is suffering from underfitting or overfitting and help us understand how model complexity impacts performance.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, validation_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import learning_curve
from sklearn.metrics import accuracy_score
 
# 1. Load and preprocess the dataset (Iris dataset for simplicity)
def load_dataset():
    data = load_iris()
    X = data.data
    y = data.target
    return X, y
 
# 2. Train a Random Forest classifier
def train_model(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model
 
# 3. Plot learning curves to diagnose underfitting/overfitting
def plot_learning_curve(model, X_train, y_train):
    """
    Plot learning curve to diagnose underfitting/overfitting.
    """
    train_sizes, train_scores, test_scores = learning_curve(
        model, X_train, y_train, cv=5, scoring='accuracy', n_jobs=-1)
    
    # Mean and standard deviation of training and testing scores
    train_mean = np.mean(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    
    # Plot the learning curves
    plt.plot(train_sizes, train_mean, label='Training Accuracy')
    plt.plot(train_sizes, test_mean, label='Cross-validation Accuracy')
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1)
    plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1)
    plt.title("Learning Curve")
    plt.xlabel("Training Size")
    plt.ylabel("Accuracy")
    plt.legend(loc='best')
    plt.show()
 
# 4. Plot validation curve to assess model complexity
def plot_validation_curve(model, X_train, y_train):
    """
    Plot validation curve to assess model complexity (e.g., number of estimators in Random Forest).
    """
    param_range = np.arange(1, 201, 10)  # Vary the number of estimators
    train_scores, test_scores = validation_curve(
        model, X_train, y_train, param_name="n_estimators", param_range=param_range, cv=5, scoring='accuracy', n_jobs=-1)
    
    # Mean and standard deviation of training and testing scores
    train_mean = np.mean(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    
    # Plot the validation curve
    plt.plot(param_range, train_mean, label='Training Accuracy')
    plt.plot(param_range, test_mean, label='Cross-validation Accuracy')
    plt.fill_between(param_range, train_mean - train_std, train_mean + train_std, alpha=0.1)
    plt.fill_between(param_range, test_mean - test_std, test_mean + test_std, alpha=0.1)
    plt.title("Validation Curve")
    plt.xlabel("Number of Estimators")
    plt.ylabel("Accuracy")
    plt.legend(loc='best')
    plt.show()
 
# 5. Example usage
X, y = load_dataset()
 
# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
 
# Train the Random Forest model
model = train_model(X_train, y_train)
 
# Evaluate the model on the test set
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Test set accuracy: {accuracy:.4f}")
 
# Plot learning curve to check for underfitting/overfitting
plot_learning_curve(model, X_train, y_train)
 
# Plot validation curve to assess model complexity
plot_validation_curve(model, X_train, y_train)