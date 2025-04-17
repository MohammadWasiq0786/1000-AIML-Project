"""
Project 757: Bayesian Neural Networks
Description:
Bayesian Neural Networks (BNNs) provide a probabilistic approach to deep learning, where we can quantify the uncertainty in the model's predictions. Unlike traditional neural networks that output a single deterministic prediction, BNNs output a distribution of predictions, reflecting the uncertainty in the model parameters and predictions. This is especially useful in applications where the confidence in the model’s predictions is critical. In this project, we will implement a Bayesian Neural Network using TensorFlow Probability to estimate uncertainty in predictions and provide probabilistic outputs.

Explanation:
Dataset Loading and Preprocessing: We load the Iris dataset and preprocess it by encoding the target labels into integers using LabelEncoder.

Bayesian Neural Network: We use TensorFlow Probability to build a Bayesian Neural Network (BNN). The model includes Flipout layers (a technique for efficient variational inference) that allow us to estimate the uncertainty in the model’s predictions.

Monte Carlo Sampling for Uncertainty: We perform Monte Carlo sampling by running multiple forward passes of the BNN (with dropout active) to generate a distribution of predictions. The mean prediction gives us the final classification, and the standard deviation (uncertainty) gives us the model’s confidence in its predictions.

Model Evaluation: The model’s accuracy is computed on the test set, and the uncertainty (as the standard deviation of predictions) is visualized to show how confident the model is in its predictions.

Visualization: We visualize the epistemic uncertainty (model uncertainty) by plotting the standard deviation of the predictions for each test sample. Higher uncertainty indicates that the model is less confident about its prediction.

This project demonstrates how Bayesian Neural Networks (BNNs) can be used for uncertainty estimation in machine learning. By using Monte Carlo sampling and Flipout layers, BNNs can quantify the uncertainty in their predictions, making them useful for high-stakes applications where understanding model confidence is critical.
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_probability as tfp
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
 
# 1. Load and preprocess the dataset (Iris dataset for simplicity)
def load_dataset():
    data = load_iris()
    X = data.data
    y = data.target
    return X, y
 
# 2. Build a Bayesian Neural Network using TensorFlow Probability
def build_bayesian_model(input_shape):
    """
    Build a Bayesian Neural Network with a Dense layer using TensorFlow Probability.
    """
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=input_shape),
        tf.keras.layers.Dense(64, activation='relu'),
        tfp.layers.DenseFlipout(64, activation='relu'),  # Bayesian layer with Flipout
        tfp.layers.DenseFlipout(3, activation='softmax')  # Output layer for 3 classes (Iris)
    ])
    
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model
 
# 3. Evaluate the model’s predictions and uncertainty
def evaluate_bayesian_model(model, X_test, y_test):
    """
    Evaluate the model on the test set and obtain predictions with uncertainty.
    """
    # Predict using Monte Carlo sampling (Multiple forward passes for uncertainty estimation)
    n_samples = 100
    predictions = np.array([model(X_test, training=True) for _ in range(n_samples)])
    
    # Calculate the mean and standard deviation (uncertainty) of the predictions
    mean_pred = predictions.mean(axis=0)
    uncertainty = predictions.std(axis=0)  # Uncertainty as standard deviation
    
    # Evaluate accuracy on the mean prediction
    y_pred = np.argmax(mean_pred, axis=1)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy on test set: {accuracy:.4f}")
    
    return mean_pred, uncertainty
 
# 4. Example usage
X, y = load_dataset()
 
# Encode target labels to integers
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
 
# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.3, random_state=42)
 
# Build and train the Bayesian Neural Network model
model = build_bayesian_model(input_shape=(4,))
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))
 
# Evaluate the Bayesian model on the test set and calculate uncertainty
mean_pred, uncertainty = evaluate_bayesian_model(model, X_test, y_test)
 
# Visualize the uncertainty of the model’s predictions
plt.figure(figsize=(10, 6))
plt.bar(range(len(uncertainty)), uncertainty.max(axis=1), color='skyblue')
plt.title('Epistemic Uncertainty (Uncertainty of the Model Predictions)')
plt.xlabel('Test Samples')
plt.ylabel('Uncertainty (Standard Deviation)')
plt.show()