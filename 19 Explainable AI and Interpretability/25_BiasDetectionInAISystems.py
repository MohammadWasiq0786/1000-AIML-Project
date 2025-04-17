"""
Project 745: Bias Detection in AI Systems
Description:
Bias detection in AI systems is crucial for ensuring that machine learning models make fair and unbiased decisions. Models trained on biased data may unintentionally perpetuate or even amplify these biases, leading to unfair outcomes, especially in sensitive applications such as hiring, criminal justice, and finance. In this project, we will explore methods for detecting and mitigating bias in machine learning models. We will demonstrate bias detection by analyzing the Iris dataset, where we will identify any potential gender or racial biases in a classification model.

Explanation:
Dataset Loading: We load the Iris dataset and preprocess it by encoding the target labels into integers. This dataset is used for training a classification model.

Model Training: A Random Forest classifier is trained on the dataset, which is then used to predict flower species based on input features.

Bias Detection: The detect_bias() function uses Fairness Indicators to assess whether the model's predictions are biased with respect to a sensitive attribute (e.g., gender, race). We calculate fairness metrics such as demographic parity and equal opportunity to measure if the model's predictions differ significantly across different groups.

Fairness Metrics Visualization: The plot_bias_detection() function visualizes the fairness metrics using a bar chart. This shows how the model’s performance varies with respect to the sensitive attribute.

Performance Evaluation: The accuracy_score function is used to evaluate the model’s performance on the test set, and the fairness metrics help identify if the model is making biased predictions.

This project provides a simple framework for detecting bias in machine learning models. By analyzing fairness indicators and visualizing them, we can assess whether the model treats different groups equally and fairly.
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
def detect_bias(model, X_test, y_test, sensitive_attribute):
    """
    Detect bias in the model's predictions with respect to the sensitive attribute.
    """
    y_pred = model.predict(X_test)
    
    # Create fairness indicators based on sensitive attribute (e.g., gender, race)
    fairness_indicators = FairnessIndicators(
        y_true=y_test,
        y_pred=y_pred,
        sensitive_attribute=sensitive_attribute,
        positive_label=1
    )
    
    # Calculate fairness metrics (e.g., demographic parity, equal opportunity)
    fairness_metrics = fairness_indicators.calculate_fairness_metrics()
    
    print(f"Fairness Metrics: {fairness_metrics}")
    
    return fairness_metrics
 
# 4. Visualize model performance with respect to fairness
def plot_bias_detection(fairness_metrics):
    """
    Visualize the fairness metrics (e.g., demographic parity, equal opportunity).
    """
    metrics = list(fairness_metrics.keys())
    values = list(fairness_metrics.values())
    
    plt.bar(metrics, values, color='skyblue')
    plt.title('Bias Detection in Model Predictions')
    plt.ylabel('Fairness Metric Value')
    plt.xlabel('Fairness Metrics')
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
 
# Detect bias in the model's predictions (assuming the sensitive attribute is gender or race)
sensitive_attribute = 'gender'  # This can be modified as needed
fairness_metrics = detect_bias(model, X_test, y_test, sensitive_attribute)
 
# Visualize the fairness metrics
plot_bias_detection(fairness_metrics)