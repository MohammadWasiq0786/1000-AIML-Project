"""
Project 782: Smart Grid Optimization
Description
Smart grids use AI to balance energy demand and supply, manage renewable inputs, and optimize distribution. This project simulates hourly power consumption, renewable generation, and grid load to predict optimal power draw decisions that minimize cost and prevent overload. We'll build a regression model that can assist in energy routing and load balancing.

This model is useful in energy dispatch systems, microgrid balancing, or real-time smart grid controllers. For real deployments, this can be integrated with smart meters, SCADA systems, or renewable energy monitoring tools.
"""


import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
 
# Simulate smart grid features: demand (kW), solar input (kW), wind input (kW), grid frequency (Hz)
np.random.seed(42)
n_samples = 1000
 
demand = np.random.normal(50, 15, n_samples)         # total energy need
solar_input = np.random.normal(20, 8, n_samples)     # variable based on sunlight
wind_input = np.random.normal(15, 6, n_samples)      # variable based on wind
grid_freq = np.random.normal(50, 0.5, n_samples)     # grid stability indicator
 
# Target: optimized power draw from main grid (want to use renewables first)
# Try to minimize draw while maintaining grid balance
draw = demand - (solar_input + wind_input) + (50 - grid_freq) * 0.5
draw = np.clip(draw, 0, None)  # no negative draw
 
# Feature matrix
X = np.stack([demand, solar_input, wind_input, grid_freq], axis=1)
y = draw
 
# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
 
# Build regression model
model = models.Sequential([
    layers.Input(shape=(4,)),
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(1)  # Output: optimal grid draw
])
 
model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=0)
 
# Evaluate
loss, mae = model.evaluate(X_test, y_test)
print(f"✅ Smart Grid Optimization Model MAE: {mae:.2f} kW")
 
# Plot predictions
predictions = model.predict(X_test).flatten()
plt.figure(figsize=(10, 4))
plt.plot(y_test[:100], label='Actual Draw')
plt.plot(predictions[:100], label='Predicted Draw')
plt.title("Smart Grid: Predicted vs Actual Power Draw")
plt.xlabel("Sample Index")
plt.ylabel("kW")
plt.legend()
plt.show()