"""
Project 751: Model Behavioral Testing
Description:
Model behavioral testing is essential for ensuring that machine learning models behave as expected under different conditions. Testing a model's behavior involves evaluating its predictions across a variety of scenarios, edge cases, and adversarial inputs. This project aims to create a framework for model behavioral testing that allows us to simulate different inputs, test edge cases, and assess how the model reacts to unexpected or challenging situations. We will implement this framework on a Random Forest classifier trained on the Iris dataset and evaluate how well the model handles various input scenarios.

Explanation:
Dataset Loading and Preprocessing: We load the Iris dataset and preprocess it by encoding the target labels (flower species) into integers using LabelEncoder.

Model Training: A Random Forest classifier is trained on the Iris dataset to predict flower species based on input features (e.g., sepal length, sepal width).

Behavioral Testing: The behavioral_testing() function evaluates the model's performance on:

Normal test cases (the actual test set).

Edge cases (unusual or extreme inputs, such as inputs with extremely high or low values).

The edge cases are designed to see how the model behaves when presented with inputs outside of the typical range or with extreme values.

Edge Case Evaluation: For edge_case_1, we input extreme values (e.g., all feature values set to 10), and for edge_case_2, we input minimum values (e.g., all feature values set to 0). We then check how the model reacts to these inputs, which helps us identify any vulnerabilities or unexpected behavior.

This framework helps in testing model robustness by evaluating how the model handles various scenarios. It helps uncover potential issues like overfitting, poor generalization, or incorrect responses to unusual inputs.
"""


import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
 
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
 
# 3. Behavioral Testing: Edge case and normal case testing
def behavioral_testing(model, X_test, y_test):
    """
    Test the model's behavior on normal and edge cases.
    """
    # Test with normal cases (actual test set)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy on normal test set: {accuracy:.4f}")
    
    # Test with edge cases (e.g., random data or extreme values)
    edge_case_1 = np.array([[10, 10, 10, 10]])  # Extreme values
    edge_case_2 = np.array([[0, 0, 0, 0]])  # Minimum values
    
    edge_case_predictions = model.predict(edge_case_1)
    edge_case_2_predictions = model.predict(edge_case_2)
    
    print(f"Prediction for edge case 1 (extreme values): {edge_case_predictions}")
    print(f"Prediction for edge case 2 (minimum values): {edge_case_2_predictions}")
 
# 4. Example usage
X, y = load_dataset()
 
# Encode target labels to integers
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
 
# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.3, random_state=42)
 
# Train the Random Forest model
model = train_model(X_train, y_train)
 
# Evaluate the model on normal and edge cases
behavioral_testing(model, X_test, y_test)