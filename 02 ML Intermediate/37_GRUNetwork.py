"""
Project 77. GRU network implementation
Description:
A GRU (Gated Recurrent Unit) network is a simplified and efficient variant of LSTM, ideal for modeling time-dependent sequences. In this project, we implement a GRU-based model to predict the next value in a univariate time series using TensorFlow/Keras.

About:
🔁 What This Project Demonstrates:
Implements a GRU network for time series prediction

Handles sequence windowing and reshaping for RNN input

Visualizes model accuracy by comparing predicted vs actual output
"""

# Install if not already: pip install tensorflow
 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense
 
# Simulated time series data (e.g., temperature or stock trend)
np.random.seed(42)
time_steps = 200
data = np.cos(np.linspace(0, 20, time_steps)) + np.random.normal(0, 0.2, time_steps)
 
# Normalize the data
df = pd.DataFrame({'Value': data})
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df)
 
# Create sequences for GRU input
def create_sequences(data, window_size):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i + window_size])
        y.append(data[i + window_size])
    return np.array(X), np.array(y)
 
window_size = 10
X, y = create_sequences(scaled_data, window_size)
 
# Split into train and test sets
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]
 
# Reshape input for GRU (samples, time steps, features)
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
 
# Build GRU model
model = Sequential([
    GRU(50, activation='tanh', input_shape=(window_size, 1)),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, epochs=20, verbose=0)
 
# Predict and inverse scale
y_pred = model.predict(X_test)
y_pred_inv = scaler.inverse_transform(y_pred)
y_test_inv = scaler.inverse_transform(y_test)
 
# Plot predictions vs actual
plt.figure(figsize=(10, 5))
plt.plot(y_test_inv, label='Actual')
plt.plot(y_pred_inv, label='Predicted', linestyle='--')
plt.title("GRU - Time Series Prediction")
plt.xlabel("Time Step")
plt.ylabel("Value")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()