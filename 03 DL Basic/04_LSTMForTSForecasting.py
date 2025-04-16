"""
Project 84. LSTM for time series forecasting
Description:
An LSTM (Long Short-Term Memory) network is excellent for learning long-term dependencies in time series data. This project demonstrates how to use an LSTM model to forecast future values based on past sequences — perfect for use cases like stock prices, weather, or energy consumption.

About:
🔮 What This Project Demonstrates:
Builds an LSTM-based time series forecaster

Learns from historical sequences to predict future values

Visualizes prediction performance using line plots
"""

# Install if not already: pip install tensorflow
 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
 
# Generate synthetic time series data (e.g., sinusoidal)
np.random.seed(42)
time_steps = 300
data = np.sin(np.linspace(0, 30, time_steps)) + np.random.normal(0, 0.2, time_steps)
 
# Scale data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data.reshape(-1, 1))
 
# Create input sequences for forecasting
def create_sequences(data, window_size):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i + window_size])
        y.append(data[i + window_size])
    return np.array(X), np.array(y)
 
window_size = 20
X, y = create_sequences(scaled_data, window_size)
 
# Train/test split
split_index = int(len(X) * 0.8)
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]
 
# Reshape for LSTM input
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
 
# Build LSTM model
model = Sequential([
    LSTM(64, activation='tanh', input_shape=(window_size, 1)),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, epochs=20, verbose=0)
 
# Predict and inverse transform
y_pred = model.predict(X_test)
y_pred_inv = scaler.inverse_transform(y_pred)
y_test_inv = scaler.inverse_transform(y_test)
 
# Plot actual vs predicted
plt.figure(figsize=(10, 5))
plt.plot(y_test_inv, label='Actual')
plt.plot(y_pred_inv, label='Predicted', linestyle='--')
plt.title("LSTM Forecasting - Time Series")
plt.xlabel("Time Step")
plt.ylabel("Value")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()