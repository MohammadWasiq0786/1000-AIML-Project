"""
Project 760: Anomaly Explanation System
Description:
An Anomaly Explanation System is designed to identify and explain unusual or unexpected behavior in a dataset. Anomalies can be critical in various domains such as fraud detection, medical diagnostics, and network security. In this project, we will develop a system that identifies anomalies in a dataset and provides explanations for why those anomalies were detected. We will use a combination of autoencoders (for anomaly detection) and SHAP (for model explanation) to create a system that can identify anomalies and explain the underlying reasons for their detection.

Explanation:
Dataset Loading and Preprocessing: The Iris dataset is loaded and preprocessed. We split it into training and testing sets, although in this case, the labels are not necessary for anomaly detection, as the goal is to identify unusual data points.

Autoencoder Model: An autoencoder is a type of neural network used for unsupervised anomaly detection. The model is trained to learn how to reconstruct input data. The reconstruction error (difference between the input and output) is used to detect anomalies — if the reconstruction error is significantly higher for a particular sample, it is considered an anomaly.

Anomaly Detection: After training the autoencoder, we compute the Mean Squared Error (MSE) between the input data and the reconstructed output. Samples with high MSE are flagged as anomalies.

SHAP for Explanation: The SHAP (SHapley Additive exPlanations) library is used to explain the anomalies. It shows which features contributed most to the anomaly detection for each sample. This provides insight into why the model flagged certain samples as anomalous.

Visualization: We use matplotlib to visualize the reconstruction error and highlight the anomalies detected by the model. The SHAP summary plot further explains which features were most influential in detecting anomalies.

This Anomaly Explanation System helps us understand not only which samples are anomalies but also why they were considered anomalies, offering greater transparency and interpretability in anomaly detection tasks.
"""

import numpy as np
import matplotlib.pyplot as plt
import shap
import tensorflow as tf
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from tensorflow.keras import layers, models
 
# 1. Load and preprocess the dataset (Iris dataset for simplicity)
def load_dataset():
    data = load_iris()
    X = data.data
    y = data.target
    return X, y
 
# 2. Build an autoencoder model for anomaly detection
def build_autoencoder(input_shape):
    """
    Build an autoencoder for anomaly detection.
    """
    input_layer = layers.Input(shape=input_shape)
    encoded = layers.Dense(64, activation='relu')(input_layer)
    encoded = layers.Dense(32, activation='relu')(encoded)
    bottleneck = layers.Dense(16, activation='relu')(encoded)
 
    decoded = layers.Dense(32, activation='relu')(bottleneck)
    decoded = layers.Dense(64, activation='relu')(decoded)
    decoded = layers.Dense(input_shape[0], activation='sigmoid')(decoded)
 
    autoencoder = models.Model(input_layer, decoded)
    autoencoder.compile(optimizer='adam', loss='mean_squared_error')
    
    return autoencoder
 
# 3. Train the autoencoder on the Iris dataset
def train_autoencoder(X_train):
    autoencoder = build_autoencoder(input_shape=(X_train.shape[1],))
    autoencoder.fit(X_train, X_train, epochs=50, batch_size=16, validation_split=0.2, shuffle=True)
    return autoencoder
 
# 4. Detect anomalies using the autoencoder
def detect_anomalies(model, X_test, threshold=0.1):
    """
    Detect anomalies based on reconstruction error (difference between input and output).
    """
    reconstructed = model.predict(X_test)
    mse = np.mean(np.square(X_test - reconstructed), axis=1)
    
    # Identify anomalies where the reconstruction error is above the threshold
    anomalies = mse > threshold
    return anomalies, mse
 
# 5. Explain anomalies using SHAP
def explain_anomalies_with_shap(model, X_test):
    """
    Use SHAP to explain which features contributed most to the anomaly detection.
    """
    explainer = shap.KernelExplainer(model.predict, X_test)
    shap_values = explainer.shap_values(X_test)
 
    # Visualize SHAP values
    shap.summary_plot(shap_values, X_test, feature_names=["sepal length", "sepal width", "petal length", "petal width"])
 
# 6. Example usage
X, y = load_dataset()
 
# Encode target labels to integers (though not needed for anomaly detection)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
 
# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.3, random_state=42)
 
# Train the autoencoder on the training data
autoencoder = train_autoencoder(X_train)
 
# Detect anomalies in the test data
anomalies, mse = detect_anomalies(autoencoder, X_test, threshold=0.1)
 
# Print anomaly detection results
print(f"Anomalies detected: {np.sum(anomalies)}")
print(f"Mean squared error for test samples: {mse[:10]}")  # Show first 10 MSE values
 
# Explain the anomalies using SHAP
explain_anomalies_with_shap(autoencoder, X_test)
 
# Visualize anomalies
plt.scatter(range(len(mse)), mse, c=anomalies, cmap='coolwarm')
plt.title("Anomaly Detection with Autoencoder")
plt.xlabel("Test Samples")
plt.ylabel("Reconstruction Error (MSE)")
plt.show()