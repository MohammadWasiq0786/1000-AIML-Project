"""
Project 758: Confidence Calibration for Neural Networks
Description:
Confidence calibration refers to the process of aligning the predicted probabilities of a machine learning model with the true likelihood of an event. In other words, a model should provide confidence scores (predicted probabilities) that match the actual frequency of an event occurring. For example, if a model predicts a probability of 0.8 for an event, that event should occur about 80% of the time. In this project, we will implement confidence calibration techniques, specifically Platt Scaling and Isotonic Regression, for a Neural Network model trained on the Iris dataset. We will use these techniques to improve the calibration of the model’s predictions and evaluate their effectiveness.

Explanation:
Dataset Loading and Preprocessing: We load the Iris dataset, which contains 150 samples of iris flowers across three species. We split it into training and testing sets using train_test_split.

Model Building: We build a simple neural network with two hidden layers using Keras. The model uses a softmax output layer for multi-class classification (since there are three species).

Calibration with Platt Scaling and Isotonic Regression:

Platt Scaling is used for logistic calibration of the model’s predicted probabilities. It fits a logistic regression model on the output of the neural network.

Isotonic Regression is a non-linear calibration technique that fits a piecewise constant function to the model’s predicted probabilities.

Both methods are implemented using CalibratedClassifierCV from Scikit-learn.

The Brier score loss is used to evaluate the calibration of the model. A lower Brier score indicates better calibration.

Calibration Curve Visualization: The plot_calibration_curve() function visualizes the calibration curve for the calibrated models. A well-calibrated model’s curve should closely match the diagonal line, which represents perfect calibration.

Model Evaluation: We calculate the accuracy of the model on the test set and then compare the Brier scores for the original model, Platt Scaling, and Isotonic Regression models. We also visualize the calibration curves for both calibration techniques.

This project demonstrates confidence calibration techniques to improve the quality of model predictions. Proper calibration ensures that the model’s predicted probabilities are trustworthy, which is essential for decision-making in many real-world applications.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, brier_score_loss
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import layers, models
 
# 1. Load and preprocess the dataset (Iris dataset for simplicity)
def load_dataset():
    data = load_iris()
    X = data.data
    y = data.target
    return X, y
 
# 2. Build a simple neural network model
def build_model(input_shape):
    """
    Build a simple feed-forward neural network model.
    """
    model = models.Sequential([
        layers.InputLayer(input_shape=input_shape),
        layers.Dense(64, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(3, activation='softmax')  # 3 classes for Iris dataset
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model
 
# 3. Calibration with Platt Scaling and Isotonic Regression
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
 
# 4. Visualize calibration curve
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
 
# Build and train the neural network model
model = build_model(input_shape=(4,))
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))
 
# Evaluate the model on the original test set
y_pred = np.argmax(model.predict(X_test), axis=1)
accuracy = accuracy_score(y_test, y_pred)
print(f"Original accuracy on test set: {accuracy:.4f}")
 
# Calibrate the model and evaluate Brier score
platt_model, isotonic_model, platt_score, isotonic_score = calibrate_model(model, X_train, y_train, X_test, y_test)
 
# Print the Brier scores of the calibrated models
print(f"Platt Scaling Brier Score: {platt_score:.4f}")
print(f"Isotonic Regression Brier Score: {isotonic_score:.4f}")
 
# Visualize the calibration curves of both models
plot_calibration_curve(platt_model, X_test, y_test, "Platt Scaling")
plot_calibration_curve(isotonic_model, X_test, y_test, "Isotonic Regression")