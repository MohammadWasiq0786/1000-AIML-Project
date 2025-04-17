"""
Project 755: Drift Adaptation Techniques
Description:
Drift adaptation refers to the strategies and techniques used to adapt machine learning models when concept drift occurs. As data distributions shift over time, models that were once accurate may become outdated, leading to degraded performance. Drift adaptation techniques help the model adapt to these changes by updating the model, retraining it, or using online learning approaches. In this project, we will implement drift adaptation techniques to address concept drift in a Random Forest classifier trained on the Iris dataset. We will use incremental learning (using online learning techniques) and model retraining to adapt to the drifted data.

Explanation:
Dataset Loading and Preprocessing: We load the Iris dataset and split it into training and testing sets using train_test_split.

Model Training: A Random Forest classifier is trained on the Iris dataset to predict flower species.

Concept Drift Simulation: The simulate_concept_drift() function simulates concept drift by altering the class distribution in the test data. A fraction of the test set is modified to simulate how the data distribution may change over time.

Drift Adaptation Techniques:

Model Retraining: When drift is detected, the model is retrained on the latest training data to adapt to the changes in the data distribution.

Online Learning (Incremental Learning): While not explicitly implemented in this code (for simplicity), online learning can be incorporated using algorithms that support incremental updates (e.g., AdaBoost or SGDClassifier). This allows the model to learn from new data as it arrives, without retraining from scratch.

Performance Evaluation: The accuracy is evaluated on the original test set, the drifted test set, and after retraining to see how well the model adapts to concept drift.

By using these drift adaptation techniques, we can keep the model up to date and improve its robustness against changes in the data distribution over time. This is especially useful in real-world applications where data is dynamic and evolving.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import AdaBoostClassifier
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
 
# 3. Concept drift simulation: Gradually change class distribution
def simulate_concept_drift(X, y, drift_percentage=0.1):
    """
    Simulate concept drift by gradually changing the class distribution.
    """
    X, y = shuffle(X, y, random_state=42)
    drifted_y = np.copy(y)
    n_samples = len(drifted_y)
    n_drift = int(n_samples * drift_percentage)  # Number of samples to apply drift to
    
    # Introduce drift by changing a fraction of the class labels
    drifted_y[-n_drift:] = (drifted_y[-n_drift:] + 1) % 3  # Change class labels to another class
    return X, drifted_y
 
# 4. Drift Adaptation: Incremental learning and model retraining
def adapt_to_drift(model, X_train, y_train, X_test, y_test, drift_percentage=0.1):
    """
    Apply drift adaptation techniques: online learning (incremental) and model retraining.
    """
    # Initial evaluation on the original test set
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Original accuracy on test set: {accuracy:.4f}")
    
    # Simulate concept drift in the test set
    X_test_drifted, y_test_drifted = simulate_concept_drift(X_test, y_test, drift_percentage)
    
    # Evaluate on the drifted data
    y_pred_drifted = model.predict(X_test_drifted)
    drifted_accuracy = accuracy_score(y_test_drifted, y_pred_drifted)
    print(f"Accuracy after concept drift: {drifted_accuracy:.4f}")
    
    # Model retraining: Update the model based on new (drifted) data
    model.fit(X_train, y_train)  # Retrain the model on the latest data
    
    # Evaluate after retraining
    y_pred_retrained = model.predict(X_test)
    retrained_accuracy = accuracy_score(y_test, y_pred_retrained)
    print(f"Accuracy after retraining: {retrained_accuracy:.4f}")
 
    return accuracy, drifted_accuracy, retrained_accuracy
 
# 5. Example usage
X, y = load_dataset()
 
# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
 
# Train the Random Forest model
model = train_model(X_train, y_train)
 
# Adapt to concept drift with drift adaptation techniques
original_accuracy, drifted_accuracy, retrained_accuracy = adapt_to_drift(model, X_train, y_train, X_test, y_test, drift_percentage=0.2)
 
# Visualize the accuracy before and after adaptation
plt.bar(["Original", "Drifted", "Retrained"], [original_accuracy, drifted_accuracy, retrained_accuracy], color=['skyblue', 'lightcoral', 'lightgreen'])
plt.title("Impact of Drift Adaptation on Model Accuracy")
plt.ylabel("Accuracy")
plt.show()