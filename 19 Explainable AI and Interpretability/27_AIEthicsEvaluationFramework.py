"""
Project 747: AI Ethics Evaluation Framework
Description:
An AI Ethics Evaluation Framework is designed to ensure that AI systems are developed and deployed in a way that aligns with ethical principles such as fairness, transparency, accountability, and non-discrimination. This framework helps organizations evaluate and mitigate the ethical risks associated with AI models, ensuring they do not perpetuate bias, harm, or unfair outcomes. In this project, we will implement an AI ethics evaluation framework for machine learning models, focusing on areas such as fairness, explainability, accountability, and non-discrimination. We will assess the model using ethical guidelines and fairness metrics, as well as interpretability tools.

Explanation:
Dataset Loading and Preprocessing: We load and preprocess the Iris dataset, which is used for classification. We encode the target labels into integers using LabelEncoder to fit them into the model.

Model Training: We train a Random Forest classifier on the Iris dataset to predict the species of flowers.

Fairness Evaluation: The evaluate_fairness() function uses Fairness Indicators to assess the fairness of the model’s predictions based on a sensitive attribute (e.g., gender, race). It calculates fairness metrics such as demographic parity and equalized odds to ensure that the model doesn't favor one group over another.

Explainability: The explain_model_with_shap() function uses SHAP (SHapley Additive exPlanations) to visualize the importance of features in the model’s predictions. This helps us understand which features the model relies on for making predictions.

Visualization: The visualize_ethics() function visualizes the fairness metrics and the model’s accuracy, allowing us to assess both performance and ethical behavior. We can check if the model performs well and fairly across different groups.

This framework allows us to evaluate the ethical implications of machine learning models, ensuring that they operate in a fair, transparent, and accountable manner.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import shap
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
 
# 3. AI Ethics Evaluation: Fairness and Bias detection
def evaluate_fairness(model, X_test, y_test, sensitive_attribute):
    """
    Evaluate fairness using fairness metrics.
    """
    y_pred = model.predict(X_test)
    
    # Fairness Indicators calculation using sensitive attributes
    fairness_indicators = FairnessIndicators(
        y_true=y_test,
        y_pred=y_pred,
        sensitive_attribute=sensitive_attribute,
        positive_label=1
    )
    
    # Calculate and return fairness metrics
    fairness_metrics = fairness_indicators.calculate_fairness_metrics()
    return fairness_metrics
 
# 4. Explainability using SHAP
def explain_model_with_shap(model, X_test):
    """
    Use SHAP to explain the model's predictions.
    """
    explainer = shap.Explainer(model)
    shap_values = explainer(X_test)
    
    # Visualize SHAP values (feature importance)
    shap.summary_plot(shap_values, X_test)
    
# 5. Visualize AI ethics (Fairness and Performance)
def visualize_ethics(fairness_metrics, accuracy):
    """
    Visualize fairness metrics and model accuracy for ethics evaluation.
    """
    # Plot fairness metrics
    metrics = list(fairness_metrics.keys())
    values = list(fairness_metrics.values())
    plt.bar(metrics, values, color='lightcoral')
    plt.title('Fairness Metrics Visualization')
    plt.xlabel('Fairness Metric')
    plt.ylabel('Metric Value')
    plt.show()
 
    # Plot model accuracy
    plt.bar(['Accuracy'], [accuracy], color='skyblue')
    plt.title('Model Performance (Accuracy)')
    plt.ylabel('Accuracy')
    plt.show()
 
# 6. Example usage
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
 
# Evaluate fairness using a sensitive attribute (e.g., gender or race)
sensitive_attribute = 'gender'  # Example sensitive attribute (can be adjusted based on dataset)
fairness_metrics = evaluate_fairness(model, X_test, y_test, sensitive_attribute)
 
# Visualize fairness metrics and model accuracy
visualize_ethics(fairness_metrics, accuracy)
 
# Explain model predictions with SHAP
explain_model_with_shap(model, X_test)