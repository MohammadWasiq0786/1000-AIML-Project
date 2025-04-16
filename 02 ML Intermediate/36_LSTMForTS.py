"""
Project 76. LSTM for time series prediction
Description:
An LSTM (Long Short-Term Memory) model is ideal for forecasting time series data where temporal dependencies matter. In this project, we use an LSTM-based neural network built with Keras to predict future values of a univariate time series (e.g., stock price, temperature).

About:
⏳ What This Project Demonstrates:
Builds an LSTM model for time series forecasting

Handles sequence generation and reshaping for LSTM input

Evaluates and visualizes predictions against actual values
"""

# Install if not already: pip install tensorflow
 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
 
# Generate synthetic time series data (e.g., sinusoidal signal)
np.random.seed(42)
time_steps = 200
data = np.sin(np.linspace(0, 20, time_steps)) + np.random.normal(0, 0.2, time_steps)
 
# Prepare dataset
df = pd.DataFrame({'Value': data})
 
# Normalize data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df)
 
# Create sequences
def create_sequences(data, window_size):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i + window_size])
        y.append(data[i + window_size])
    return np.array(X), np.array(y)
 
window_size = 10
X, y = create_sequences(scaled_data, window_size)
 
# Train/test split
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]
 
# Reshape for LSTM input (samples, timesteps, features)
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
 
# Build LSTM model
model = Sequential([
    LSTM(50, activation='relu', input_shape=(window_size, 1)),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, epochs=20, verbose=0)
 
# Predict and inverse scale
y_pred = model.predict(X_test)
y_test_inv = scaler.inverse_transform(y_test)
y_pred_inv = scaler.inverse_transform(y_pred)
 
# Plot actual vs predicted
plt.figure(figsize=(10, 5))
plt.plot(y_test_inv, label='Actual')
plt.plot(y_pred_inv, label='Predicted', linestyle='--')
plt.title("LSTM - Time Series Prediction")
plt.xlabel("Time Steps")
plt.ylabel("Value")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()