"""
Project 728: Counterfactual Explanations Generator
Description:
Counterfactual explanations help explain the decisions of a machine learning model by presenting what would have happened if the input features had been different. Essentially, a counterfactual explanation tells us how the model’s prediction would change if we changed certain aspects of the input. This is useful for understanding model behavior and for debugging. In this project, we will implement a counterfactual explanation generator that creates counterfactual examples by perturbing the features of an input instance and seeing how the model’s output changes.

Explanation:
Dataset and Preprocessing: We load the Iris dataset and preprocess it using MinMaxScaler to normalize the feature values between 0 and 1. This normalization is important for the counterfactual generation process since we perturb features within a bounded range.

Model Training: We train a Random Forest classifier on the Iris dataset, which is a simple yet powerful model for classification tasks.

Counterfactual Generation: The generate_counterfactual() function perturbs the input features of a sample until a different prediction is made by the model. The perturbation is done by randomly adjusting the values of the features (within a small range defined by epsilon) until the model's output changes.

Visualization: The visualize_counterfactual() function displays the original instance and its counterfactual counterpart side by side, highlighting the feature changes.

This technique can be used for any classification model to understand how small changes to input features influence the model's predictions, providing explainability and interpretability.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
 
# 1. Load and preprocess the dataset (Iris dataset for simplicity)
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
 
# 3. Generate counterfactual explanation
def generate_counterfactual(model, instance, X_train, y_train, epsilon=0.1):
    """
    Generate a counterfactual by modifying the instance features until a different prediction is made.
    """
    # Get the original prediction
    original_pred = model.predict([instance])[0]
    
    # Normalize the instance using the MinMaxScaler based on the training data
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    instance_scaled = scaler.transform([instance])[0]
    
    # Start perturbing the instance to generate a counterfactual
    counterfactual = np.copy(instance_scaled)
    while model.predict([counterfactual])[0] == original_pred:
        # Randomly perturb the features within a range (epsilon)
        feature_to_change = np.random.randint(0, len(instance))
        perturbation = np.random.uniform(-epsilon, epsilon)
        counterfactual[feature_to_change] = min(max(counterfactual[feature_to_change] + perturbation, 0), 1)
    
    # Rescale back to the original scale
    counterfactual_rescaled = scaler.inverse_transform([counterfactual])
    return counterfactual_rescaled[0]
 
# 4. Visualize original instance and counterfactual
def visualize_counterfactual(original, counterfactual, feature_names):
    """
    Visualize the original instance and counterfactual instance side by side.
    """
    plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 1)
    plt.bar(feature_names, original, color='b')
    plt.title("Original Instance")
    
    plt.subplot(1, 2, 2)
    plt.bar(feature_names, counterfactual, color='r')
    plt.title("Counterfactual Instance")
    
    plt.show()
 
# 5. Example usage
X, y, feature_names = load_dataset()
 
# Split dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
 
# Train the Random Forest model
model = train_model(X_train, y_train)
 
# Select an instance to generate a counterfactual explanation for
instance = X_test[0]  # Use the first test instance
 
# Generate a counterfactual for the instance
counterfactual_instance = generate_counterfactual(model, instance, X_train, y_train)
 
# Visualize the original and counterfactual instances
visualize_counterfactual(instance, counterfactual_instance, feature_names)