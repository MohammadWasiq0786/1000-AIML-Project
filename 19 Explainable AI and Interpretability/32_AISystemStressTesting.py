"""
Project 752: AI System Stress Testing
Description:
AI system stress testing is the process of evaluating how well an AI system performs under extreme conditions or unexpected inputs. Stress testing helps identify potential weaknesses, limitations, and failure points in the system. This can include testing the model with large volumes of data, adversarial inputs, or highly imbalanced data. In this project, we will implement stress testing for an AI system, specifically a machine learning model, by simulating scenarios such as data overload, adversarial attacks, and handling edge cases.

Explanation:
Dataset Loading and Preprocessing: We load the Iris dataset and preprocess it. This dataset contains flower species data, and we split it into training and testing sets.

Model Training: We train a Random Forest classifier on the Iris dataset to predict the species of flowers.

Adversarial Testing: The adversarial_testing() function adds random noise to the test set to simulate adversarial inputs (inputs intentionally designed to fool the model). We evaluate the model's performance on these noisy inputs, helping to identify how robust the model is to slight perturbations.

Imbalanced Data Testing: The imbalanced_data_testing() function creates an imbalanced dataset by retaining only one class from the training set. This simulates a real-world scenario where the data might be highly skewed, and the model is forced to train on imbalanced data. We then evaluate the model on the original test set to assess its performance.

Performance Evaluation: We print the accuracy of the model on the original test set, the adversarial test set (with added noise), and the imbalanced training data.

This stress testing framework helps identify potential vulnerabilities in a model, such as sensitivity to noise (adversarial attacks) or poor generalization when trained on imbalanced data.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
 
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
 
# 3. Stress Testing with Adversarial Inputs
def adversarial_testing(model, X_test, y_test):
    """
    Test the model's robustness to adversarial inputs by adding noise to the test data.
    """
    noise = np.random.normal(0, 0.1, X_test.shape)  # Adding small random noise
    X_test_adversarial = X_test + noise
    
    # Predict with adversarial inputs
    y_pred = model.predict(X_test_adversarial)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy after adversarial attack: {accuracy:.4f}")
    
    return accuracy
 
# 4. Stress Testing with Imbalanced Data
def imbalanced_data_testing(model, X_train, y_train, X_test, y_test):
    """
    Simulate a stress test with imbalanced data (creating a class imbalance in the dataset).
    """
    # Make the dataset highly imbalanced by keeping only one class
    X_train_imbalanced = X_train[y_train == 0]  # Keep only class 0
    y_train_imbalanced = y_train[y_train == 0]
    
    # Re-train the model on imbalanced data
    model.fit(X_train_imbalanced, y_train_imbalanced)
    
    # Evaluate model on original test set
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy after training on imbalanced data: {accuracy:.4f}")
    
    return accuracy
 
# 5. Example usage
X, y = load_dataset()
 
# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
 
# Train the Random Forest model
model = train_model(X_train, y_train)
 
# Evaluate the model on the original test set
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Original accuracy on test set: {accuracy:.4f}")
 
# Stress test with adversarial inputs (adding noise to the test data)
adversarial_accuracy = adversarial_testing(model, X_test, y_test)
 
# Stress test with imbalanced training data (class imbalance)
imbalanced_accuracy = imbalanced_data_testing(model, X_train, y_train, X_test, y_test)