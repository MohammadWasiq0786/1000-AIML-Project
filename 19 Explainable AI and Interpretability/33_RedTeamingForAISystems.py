"""
Project 753: Red Teaming for AI Systems
Description:
Red teaming is a method of stress testing AI systems by simulating adversarial attacks, unexpected inputs, and other scenarios to evaluate their robustness, security, and performance. The goal of red teaming is to uncover potential vulnerabilities and weaknesses in AI models that could be exploited or cause failures in real-world situations. In this project, we will implement a Red Teaming approach for a machine learning model by simulating adversarial inputs and testing the model's resilience to these attacks. We will use the Iris dataset and a Random Forest classifier to demonstrate this concept.

Explanation:
Dataset Loading and Preprocessing: We load the Iris dataset and preprocess it by encoding the target labels into integers using LabelEncoder.

Model Training: We train a Random Forest classifier on the Iris dataset to classify flower species based on the input features (e.g., sepal length, sepal width).

Adversarial Attack Simulation: The adversarial_attack() function simulates an adversarial attack by adding random noise to the input data. This noise is small (controlled by epsilon) but can significantly affect the model's predictions. This represents how an attacker might perturb the inputs to confuse the model.

Red Teaming Evaluation: The red_team_testing() function evaluates the model's performance on both the original and adversarial test sets. It prints the accuracy on both sets, allowing us to compare how the model performs in normal conditions versus under adversarial attack.

Performance Evaluation: The model's accuracy on the original test set is compared with its accuracy on the adversarial test set to assess how well the model handles adversarial inputs.

This project demonstrates Red Teaming for evaluating the robustness of machine learning models. It helps identify potential vulnerabilities in models by simulating adversarial attacks, which are a key concern in real-world AI deployments.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
 
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
 
# 3. Simulate adversarial attack (perturbing the input data)
def adversarial_attack(X_test, epsilon=0.1):
    """
    Simulate an adversarial attack by adding small perturbations (noise) to the input data.
    """
    perturbation = np.random.normal(0, epsilon, X_test.shape)  # Add random noise
    X_test_adversarial = X_test + perturbation
    return X_test_adversarial
 
# 4. Red Teaming: Evaluate model's robustness to adversarial inputs
def red_team_testing(model, X_test, y_test, epsilon=0.1):
    """
    Perform red teaming by simulating adversarial attacks on the model's test set.
    """
    # Evaluate accuracy on original test set
    y_pred = model.predict(X_test)
    original_accuracy = accuracy_score(y_test, y_pred)
    print(f"Original test set accuracy: {original_accuracy:.4f}")
 
    # Simulate adversarial attack
    X_test_adversarial = adversarial_attack(X_test, epsilon)
 
    # Evaluate accuracy on adversarial test set
    y_pred_adversarial = model.predict(X_test_adversarial)
    adversarial_accuracy = accuracy_score(y_test, y_pred_adversarial)
    print(f"Accuracy after adversarial attack: {adversarial_accuracy:.4f}")
 
# 5. Example usage
X, y = load_dataset()
 
# Encode target labels to integers
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
 
# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.3, random_state=42)
 
# Train the Random Forest model
model = train_model(X_train, y_train)
 
# Red Teaming: Evaluate the model's robustness to adversarial inputs
red_team_testing(model, X_test, y_test, epsilon=0.1)