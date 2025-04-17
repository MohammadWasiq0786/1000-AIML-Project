"""
Project 746: Fairness Evaluation Toolkit
Description:
A Fairness Evaluation Toolkit is essential for evaluating and ensuring that machine learning models make fair and unbiased decisions, especially when deployed in sensitive applications such as hiring, lending, and law enforcement. In this project, we will build a toolkit that calculates various fairness metrics, such as demographic parity, equalized odds, and disparate impact, to evaluate how a model performs across different subgroups (e.g., based on gender, race, or age). We will implement the toolkit to assess the fairness of a Random Forest classifier on the Iris dataset, which will allow us to evaluate the fairness of the model's predictions.

Explanation:
Dataset Loading and Preprocessing: We load the Iris dataset and preprocess it by encoding the target labels (species) into integers using LabelEncoder. This allows us to train the model for classification tasks.

Model Training: A Random Forest classifier is used to train a model on the preprocessed dataset. We use 100 estimators (trees) in the forest.

Fairness Metrics Calculation: The fairness_metrics() function uses the Fairness Indicators package to calculate fairness metrics such as demographic parity, equalized odds, and disparate impact. The fairness metrics help us determine whether the model’s predictions differ significantly across different groups based on a sensitive attribute (e.g., gender, race).

Visualization: The plot_fairness_metrics() function visualizes the calculated fairness metrics using matplotlib to help identify whether the model is biased toward any particular group.

Performance Evaluation: The model’s accuracy is calculated on the test set using accuracy_score, and fairness metrics are then calculated and visualized.

This toolkit helps in evaluating the fairness of machine learning models, providing insights into whether the model's performance varies based on sensitive attributes like gender or race.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from fairness_indicators import FairnessIndicators
 
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
 
# 3. Fairness Metrics: Bias detection in model predictions
def fairness_metrics(model, X_test, y_test, sensitive_attribute):
    """
    Evaluate fairness metrics such as demographic parity, equalized odds, and disparate impact.
    """
    y_pred = model.predict(X_test)
    
    # Fairness Indicators calculation using sensitive attributes
    fairness_indicators = FairnessIndicators(
        y_true=y_test,
        y_pred=y_pred,
        sensitive_attribute=sensitive_attribute,
        positive_label=1
    )
    
    # Calculate and return the fairness metrics
    fairness_metrics = fairness_indicators.calculate_fairness_metrics()
    return fairness_metrics
 
# 4. Visualize fairness metrics
def plot_fairness_metrics(fairness_metrics):
    """
    Visualize fairness metrics (e.g., demographic parity, equalized odds, etc.)
    """
    metrics = list(fairness_metrics.keys())
    values = list(fairness_metrics.values())
    
    plt.bar(metrics, values, color='lightcoral')
    plt.title('Fairness Metrics Visualization')
    plt.xlabel('Fairness Metric')
    plt.ylabel('Metric Value')
    plt.show()
 
# 5. Example usage
X, y, feature_names = load_dataset()
 
# Encode target labels to integers
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
 
# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.3, random_state=42)
 
# Train the Random Forest model
model = train_model(X_train, y_train)
 
# Evaluate the model on the test set
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Test set accuracy: {accuracy:.4f}")
 
# Evaluate fairness metrics based on the sensitive attribute
sensitive_attribute = 'gender'  # Example sensitive attribute
fairness_metrics = fairness_metrics(model, X_test, y_test, sensitive_attribute)
 
# Visualize the fairness metrics
plot_fairness_metrics(fairness_metrics)