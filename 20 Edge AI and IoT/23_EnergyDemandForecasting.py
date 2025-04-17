"""
Project 783: Energy Demand Forecasting
Description
Energy demand forecasting predicts future electricity usage to enable proactive load balancing, reduce operational costs, and prevent blackouts. Accurate forecasting is essential for utility companies and smart grids. In this project, we simulate daily energy demand patterns and build an LSTM time-series model to predict the next day’s demand.

This type of model can be expanded for hourly or 15-minute intervals, and applied in smart grid load planning, utility billing predictions, or HVAC automation.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
 
# Simulate daily energy consumption data (kWh)
np.random.seed(42)
days = 365  # 1 year of data
base_pattern = 50 + 10 * np.sin(np.arange(days) * 2 * np.pi / 30)  # monthly cycles
noise = np.random.normal(0, 2, days)
demand = base_pattern + noise
demand = demand.reshape(-1, 1)
 
# Normalize the demand values
scaler = MinMaxScaler()
demand_scaled = scaler.fit_transform(demand)
 
# Prepare LSTM sequences
def create_sequences(data, window_size=7):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i+window_size])
        y.append(data[i+window_size])
    return np.array(X), np.array(y)
 
window_size = 7  # one week window
X, y = create_sequences(demand_scaled, window_size)
 
# Train-test split
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]
 
# Build LSTM model
model = models.Sequential([
    layers.LSTM(32, input_shape=(window_size, 1)),
    layers.Dense(16, activation='relu'),
    layers.Dense(1)
])
 
model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, epochs=20, batch_size=16, verbose=0)
 
# Predict and denormalize
predictions = model.predict(X_test)
predicted_demand = scaler.inverse_transform(predictions)
actual_demand = scaler.inverse_transform(y_test)
 
# Plot forecast results
plt.figure(figsize=(10, 4))
plt.plot(actual_demand[:50], label="Actual")
plt.plot(predicted_demand[:50], label="Predicted")
plt.title("Energy Demand Forecasting")
plt.xlabel("Day")
plt.ylabel("kWh")
plt.legend()
plt.show()
 
print("✅ LSTM model trained for daily energy demand forecasting.")