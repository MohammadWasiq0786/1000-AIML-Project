"""
Project 846: Wind Power Generation Prediction
Description
Wind power generation prediction enables better planning and integration of wind energy into the grid. It depends on factors like wind speed, air density, blade length, and temperature. In this project, we simulate relevant inputs and build a regression model to estimate wind turbine power output in kilowatts (kW).

This model supports:

Wind turbine monitoring

Grid supply forecasting

Hybrid solar-wind energy management systems
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
 
# Simulate input features
np.random.seed(42)
n_samples = 1000
 
wind_speed = np.random.normal(8, 2, n_samples)                # m/s
air_density = np.random.normal(1.225, 0.05, n_samples)        # kg/m³ (standard at sea level)
blade_length = np.random.normal(40, 5, n_samples)             # meters
temperature = np.random.normal(15, 5, n_samples)              # °C
 
# Wind power formula (simplified): P = 0.5 * ρ * A * v³
swept_area = np.pi * blade_length**2
power_output = 0.5 * air_density * swept_area * wind_speed**3 * 0.4  # assuming 40% efficiency
power_output = power_output / 1000  # convert to kW
power_output += np.random.normal(0, 100, n_samples)  # add noise
 
# Feature matrix and labels
X = np.stack([wind_speed, air_density, blade_length, temperature], axis=1)
y = power_output
 
# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
 
# Build regression model
model = models.Sequential([
    layers.Input(shape=(4,)),
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(1)  # Output: wind power in kW
])
 
model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=0)
 
# Evaluate model
loss, mae = model.evaluate(X_test, y_test)
print(f"✅ Wind Power Prediction MAE: {mae:.2f} kW")
 
# Predict and plot results
preds = model.predict(X_test[:50]).flatten()
plt.figure(figsize=(10, 4))
plt.plot(y_test[:50], label="Actual Power")
plt.plot(preds, label="Predicted Power")
plt.title("Wind Power Generation Prediction")
plt.xlabel("Sample Index")
plt.ylabel("Power Output (kW)")
plt.legend()
plt.grid(True)
plt.show()