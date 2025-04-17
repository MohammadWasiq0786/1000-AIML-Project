"""
Project 749: Trust Calibration in AI Systems
Description:
Trust calibration refers to the process of aligning the confidence (predicted probability) of a machine learning model with its actual performance. A well-calibrated model’s predicted probabilities should closely match the true likelihood of an event. For example, if a model predicts a probability of 0.8 for an event, that event should occur approximately 80% of the time. This project will focus on evaluating and improving the calibration of a machine learning model. Specifically, we will use calibration techniques like Platt Scaling and Isotonic Regression to calibrate a Random Forest classifier and evaluate the calibration performance using Brier score.

Explanation:
Dataset Loading and Preprocessing: We load the Iris dataset and preprocess it by encoding the target labels (species) into integers using LabelEncoder.

Model Training: A Random Forest classifier is trained on the dataset using 100 trees in the forest.

Model Calibration:

We calibrate the trained model using Platt Scaling and Isotonic Regression, both of which are common calibration methods:

Platt Scaling uses a logistic regression model to map predicted probabilities to calibrated values.

Isotonic Regression applies a non-linear function to map predicted probabilities to calibrated values.

Brier score loss is used to evaluate how well the calibrated models predict probabilities. A lower Brier score indicates better calibration.

Calibration Curve Visualization: We use calibration_curve to plot the calibration curve, which shows how well the predicted probabilities align with the actual outcomes. A well-calibrated model should have a curve that closely matches the diagonal line (perfect calibration).

Model Evaluation: We print the Brier score loss for both Platt Scaling and Isotonic Regression models and visualize their calibration curves to assess the quality of calibration.

By using these calibration methods, we improve the model’s trustworthiness and ensure that its predicted probabilities are reliable. This is crucial when the model’s predictions are used in decision-making processes.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import brier_score_loss
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
 
# 3. Calibrate the model using Platt Scaling and Isotonic Regression
def calibrate_model(model, X_train, y_train, X_test, y_test):
    """
    Calibrate the model using Platt Scaling and Isotonic Regression.
    """
    # Platt scaling: Logistic calibration
    platt_model = CalibratedClassifierCV(model, method='sigmoid', cv='prefit')
    platt_model.fit(X_train, y_train)
 
    # Isotonic regression: Non-linear calibration
    isotonic_model = CalibratedClassifierCV(model, method='isotonic', cv='prefit')
    isotonic_model.fit(X_train, y_train)
 
    # Evaluate the calibration performance using Brier score loss
    platt_score = brier_score_loss(y_test, platt_model.predict_proba(X_test)[:, 1])
    isotonic_score = brier_score_loss(y_test, isotonic_model.predict_proba(X_test)[:, 1])
 
    return platt_model, isotonic_model, platt_score, isotonic_score
 
# 4. Visualize the calibration curve
def plot_calibration_curve(model, X_test, y_test, model_name):
    """
    Plot the calibration curve to visualize model calibration.
    """
    prob_true, prob_pred = calibration_curve(y_test, model.predict_proba(X_test)[:, 1], n_bins=10)
    
    plt.plot(prob_pred, prob_true, marker='o', label=f'{model_name} Calibration Curve')
    plt.plot([0, 1], [0, 1], linestyle='--', label='Perfect Calibration')
    plt.title(f'{model_name} Calibration Curve')
    plt.xlabel('Mean Predicted Probability')
    plt.ylabel('Fraction of Positives')
    plt.legend()
    plt.show()
 
# 5. Example usage
X, y = load_dataset()
 
# Encode target labels to integers
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
 
# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.3, random_state=42)
 
# Train the Random Forest model
model = train_model(X_train, y_train)
 
# Calibrate the model and evaluate Brier score
platt_model, isotonic_model, platt_score, isotonic_score = calibrate_model(model, X_train, y_train, X_test, y_test)
 
# Print the Brier score of the calibrated models
print(f"Platt Scaling Brier Score: {platt_score:.4f}")
print(f"Isotonic Regression Brier Score: {isotonic_score:.4f}")
 
# Visualize the calibration curves of both models
plot_calibration_curve(platt_model, X_test, y_test, "Platt Scaling")
plot_calibration_curve(isotonic_model, X_test, y_test, "Isotonic Regression")