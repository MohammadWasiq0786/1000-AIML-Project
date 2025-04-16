"""
Project 83. RNN for sequence prediction
Description:
A Recurrent Neural Network (RNN) is a type of neural network designed for sequential data such as time series, language, or sensor readings. In this project, we build a basic RNN using Keras to predict the next value in a numerical sequence based on past observations.

About:
🔁 What This Project Demonstrates:
Implements a Simple RNN for predicting sequential values

Prepares sliding window sequences from a time series

Trains and visualizes predictions vs actual values
"""

# Install if not already: pip install tensorflow
 
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense
from sklearn.preprocessing import MinMaxScaler
 
# Generate synthetic sequence data (e.g., sine wave)
np.random.seed(42)
data_points = 200
t = np.linspace(0, 20, data_points)
data = np.sin(t) + np.random.normal(0, 0.1, data_points)
 
# Normalize data
scaler = MinMaxScaler()
data = scaler.fit_transform(data.reshape(-1, 1))
 
# Create sequences
def create_sequences(data, window_size):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i+window_size])
        y.append(data[i+window_size])
    return np.array(X), np.array(y)
 
window_size = 10
X, y = create_sequences(data, window_size)
 
# Split into train and test sets
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]
 
# Reshape input for RNN (samples, timesteps, features)
X_train = X_train.reshape((X_train.shape[0], window_size, 1))
X_test = X_test.reshape((X_test.shape[0], window_size, 1))
 
# Build the RNN model
model = Sequential([
    SimpleRNN(50, activation='tanh', input_shape=(window_size, 1)),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, epochs=20, verbose=0)
 
# Predict
y_pred = model.predict(X_test)
y_test_inv = scaler.inverse_transform(y_test)
y_pred_inv = scaler.inverse_transform(y_pred)
 
# Plot actual vs predicted
plt.figure(figsize=(10, 5))
plt.plot(y_test_inv, label='Actual')
plt.plot(y_pred_inv, label='Predicted', linestyle='--')
plt.title("RNN Sequence Prediction")
plt.xlabel("Time Step")
plt.ylabel("Value")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()