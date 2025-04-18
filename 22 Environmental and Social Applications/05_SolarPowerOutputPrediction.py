"""
Project 845: Solar Power Output Prediction
Description
Solar power output prediction helps manage energy generation, storage, and grid integration. It depends heavily on weather factors like solar irradiance, cloud cover, and temperature. In this project, we simulate these inputs and build a regression model to predict solar energy output (in kWh) from a solar panel array.

This model is useful for:

Off-grid solar management systems

Solar farm planning tools

Smart home energy dashboards
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
 
# Simulate solar input features
np.random.seed(42)
n_samples = 1000
 
solar_irradiance = np.random.normal(700, 100, n_samples)      # W/m²
temperature = np.random.normal(30, 5, n_samples)               # °C
cloud_cover = np.random.uniform(0, 1, n_samples)               # 0 (clear) to 1 (fully cloudy)
panel_efficiency = np.random.normal(0.18, 0.01, n_samples)     # % efficiency
 
# Simulate solar energy output (kWh)
# Output decreases with cloud cover and excessive heat
solar_output = (
    solar_irradiance * panel_efficiency * (1 - 0.6 * cloud_cover) -
    0.05 * np.maximum(temperature - 35, 0) +
    np.random.normal(0, 5, n_samples)
) / 100  # Scale down to kWh for a single panel
 
# Feature matrix
X = np.stack([solar_irradiance, temperature, cloud_cover, panel_efficiency], axis=1)
y = solar_output
 
# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
 
# Build the regression model
model = models.Sequential([
    layers.Input(shape=(4,)),
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(1)  # Predicted solar output in kWh
])
 
model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=0)
 
# Evaluate model
loss, mae = model.evaluate(X_test, y_test)
print(f"✅ Solar Output Prediction MAE: {mae:.2f} kWh")
 
# Plot predictions
preds = model.predict(X_test[:50]).flatten()
plt.figure(figsize=(10, 4))
plt.plot(y_test[:50], label="Actual Output")
plt.plot(preds, label="Predicted Output")
plt.title("Solar Power Output Prediction")
plt.xlabel("Sample Index")
plt.ylabel("Output (kWh)")
plt.legend()
plt.grid(True)
plt.show()