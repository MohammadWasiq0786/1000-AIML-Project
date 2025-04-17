"""
Project 741: Interpretable Time Series Models
Description:
Interpretable time series models are models that allow us to understand the influence of different time-dependent features on predictions or decisions. In time series analysis, it is important to not only predict future values but also explain how various features contribute to these predictions. This can help in understanding patterns, trends, and seasonality in the data. In this project, we will implement an interpreter for time series models that explains the model's behavior using techniques such as feature importance, shapley values, and partial dependence plots (PDP).

We will use a Long Short-Term Memory (LSTM) network for predicting future values in a time series dataset (e.g., sinusoidal wave), and then we will interpret the model’s predictions using SHAP (SHapley Additive exPlanations). SHAP provides a way to measure the importance of each feature in the prediction.

Explanation:
Synthetic Time Series Generation: The generate_time_series_data() function generates a simple sinusoidal time series pattern. This serves as an example for predicting future values based on past observations.

Data Preprocessing: The preprocess_data() function normalizes the input data using MinMaxScaler and reshapes it to fit the input requirements of LSTM (3D input).

LSTM Model: The build_lstm_model() function creates a simple LSTM model with 64 units in the hidden layer and a dense output layer. The model is trained on the time series data to predict the next value in the sequence.

SHAP for Interpretation: We use SHAP (SHapley Additive exPlanations) to explain the model's predictions. The KernelExplainer is used to approximate the importance of each feature (i.e., the time steps) in the model's decision-making process. The shap_values are then visualized using a summary plot, which shows the contribution of each feature to the prediction for all instances.

Visualization: The shap.summary_plot() function visualizes the SHAP values, which helps us interpret the effect of each time step on the model’s predictions.

By using SHAP, we can gain insights into which time steps are most influential in the model's predictions, making the LSTM model more interpretable.
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import shap
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
 
# 1. Create a synthetic time series dataset (e.g., sinusoidal wave)
def generate_time_series_data(n_samples=1000, time_steps=50):
    """
    Generate a synthetic time series dataset with a sinusoidal pattern.
    """
    t = np.linspace(0, 100, n_samples)
    y = np.sin(t)  # Sinusoidal function as the time series target
    X = np.array([y[i:i+time_steps] for i in range(n_samples - time_steps)])
    y = y[time_steps:]
    return X, y
 
# 2. Preprocess the data (Scaling and reshaping for LSTM)
def preprocess_data(X, y):
    """
    Scale the input data and reshape it for LSTM input.
    """
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_scaled = scaler.fit_transform(X)
    X_scaled = X_scaled.reshape((X_scaled.shape[0], X_scaled.shape[1], 1))  # LSTM expects 3D input
    return X_scaled, y
 
# 3. Build and train the LSTM model
def build_lstm_model(input_shape):
    """
    Build a simple LSTM model for time series prediction.
    """
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(64, activation='relu', input_shape=input_shape),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model
 
# 4. Explain the model using SHAP
def explain_model_with_shap(model, X_train):
    """
    Use SHAP to explain the model's predictions.
    """
    explainer = shap.KernelExplainer(model.predict, X_train[:100])  # Use a subset for the explainer
    shap_values = explainer.shap_values(X_train[:100])  # Calculate SHAP values for the first 100 samples
    return shap_values
 
# 5. Visualize SHAP values
def visualize_shap_values(shap_values, feature_names):
    """
    Visualize the SHAP values using summary plots.
    """
    shap.summary_plot(shap_values, feature_names=feature_names)
 
# 6. Example usage
X, y = generate_time_series_data()
 
# Preprocess data for LSTM model
X_scaled, y_scaled = preprocess_data(X, y)
 
# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)
 
# Build and train the LSTM model
model = build_lstm_model(input_shape=(X_train.shape[1], 1))
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))
 
# Explain the model's predictions with SHAP
shap_values = explain_model_with_shap(model, X_train)
 
# Visualize the SHAP values
visualize_shap_values(shap_values, feature_names=["Previous Time Step Feature"])