"""
Project 754: Concept Drift Detection
Description:
Concept drift refers to the phenomenon where the statistical properties of the target variable, or the relationship between input features and the target, change over time. This can happen in real-world applications like fraud detection, stock market predictions, or weather forecasting. When concept drift occurs, models trained on historical data may become outdated, and their performance may degrade. In this project, we will implement a system to detect and handle concept drift in a machine learning model. We will use the Iris dataset and a Random Forest classifier, and simulate concept drift by gradually changing the class distribution in the dataset.

Explanation:
Dataset Loading and Preprocessing: The Iris dataset is loaded, and the data is split into training and testing sets using train_test_split.

Model Training: A Random Forest classifier is trained on the training set of the Iris dataset.

Concept Drift Simulation: The simulate_concept_drift() function introduces concept drift by changing the class distribution of the target variable. A percentage of the data points in the test set have their class labels altered, simulating real-world changes in the data distribution over time.

Performance Evaluation: The evaluate_with_concept_drift() function evaluates the model’s performance on the original and drifted test sets. The model’s accuracy is calculated for both cases, helping us understand how well it adapts to the changes in the data.

Visualization: A bar chart is displayed showing the accuracy of the model before and after concept drift. This allows us to visually assess how concept drift impacts the model's performance.

By using this approach, we can simulate concept drift in time-dependent applications and evaluate how well the model adapts to changing data distributions. This is crucial for models deployed in dynamic environments, such as financial markets or medical diagnosis systems, where the relationship between input features and the target can change over time.
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
 
# 3. Simulate concept drift (Gradually change class distribution in the dataset)
def simulate_concept_drift(X, y, drift_percentage=0.1):
    """
    Simulate concept drift by gradually changing the class distribution.
    """
    # Shuffle the dataset to introduce randomness
    X, y = shuffle(X, y, random_state=42)
    
    # Apply concept drift by modifying the class distribution
    drifted_y = np.copy(y)
    n_samples = len(drifted_y)
    n_drift = int(n_samples * drift_percentage)  # Number of samples to apply drift to
    
    # Introduce drift by changing a fraction of the class labels
    drifted_y[-n_drift:] = (drifted_y[-n_drift:] + 1) % 3  # Change class labels to another class
    return X, drifted_y
 
# 4. Evaluate model performance over time with and without drift
def evaluate_with_concept_drift(model, X_train, y_train, X_test, y_test, drift_percentage=0.1):
    """
    Train the model on the original data and evaluate it on both the original and drifted data.
    """
    # Train the model on the original data
    model.fit(X_train, y_train)
    
    # Evaluate on the original test set
    y_pred = model.predict(X_test)
    original_accuracy = accuracy_score(y_test, y_pred)
    print(f"Original accuracy on test set: {original_accuracy:.4f}")
    
    # Simulate concept drift and evaluate on the drifted test set
    X_test_drifted, y_test_drifted = simulate_concept_drift(X_test, y_test, drift_percentage)
    y_pred_drifted = model.predict(X_test_drifted)
    drifted_accuracy = accuracy_score(y_test_drifted, y_pred_drifted)
    print(f"Accuracy after concept drift: {drifted_accuracy:.4f}")
 
    return original_accuracy, drifted_accuracy
 
# 5. Example usage
X, y = load_dataset()
 
# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
 
# Train the Random Forest model
model = train_model(X_train, y_train)
 
# Evaluate the model with concept drift
original_accuracy, drifted_accuracy = evaluate_with_concept_drift(model, X_train, y_train, X_test, y_test, drift_percentage=0.2)
 
# Visualize the impact of concept drift
plt.bar(["Original", "Drifted"], [original_accuracy, drifted_accuracy], color=['skyblue', 'lightcoral'])
plt.title("Impact of Concept Drift on Model Accuracy")
plt.ylabel("Accuracy")
plt.show()