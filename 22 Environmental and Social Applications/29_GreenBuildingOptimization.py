"""
Project 869: Green Building Optimization
Description
Green building optimization focuses on reducing energy consumption, water usage, and emissions in architecture. In this project, we simulate building features and build a multi-output regression model to estimate annual energy usage, water consumption, and carbon emissions, helping identify efficiency improvements.

✅ This model supports:

LEED/BREEAM score simulations

Architectural design feedback loops

Carbon-neutral building initiatives
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
 
# Simulate building parameters
np.random.seed(42)
n_samples = 1000
 
floor_area = np.random.normal(250, 75, n_samples)             # m²
window_ratio = np.random.uniform(0.1, 0.6, n_samples)         # % of window area
insulation_score = np.random.uniform(0, 1, n_samples)         # 0–1
occupancy_rate = np.random.normal(0.75, 0.15, n_samples)      # 0–1
appliance_efficiency = np.random.uniform(0.5, 1, n_samples)   # 0.5–1.0
 
# Outputs to predict:
# - Annual energy use (kWh)
# - Water consumption (liters)
# - Carbon emissions (kg CO₂e)
 
energy_use = (
    floor_area * 20 * (1 - insulation_score) +
    5000 * (1 - appliance_efficiency) +
    np.random.normal(0, 500, n_samples)
)
 
water_use = (
    floor_area * 10 * occupancy_rate +
    np.random.normal(0, 200, n_samples)
)
 
carbon_emissions = energy_use * 0.45 + water_use * 0.002 + np.random.normal(0, 50, n_samples)
 
# Feature matrix and target outputs
X = np.stack([floor_area, window_ratio, insulation_score, occupancy_rate, appliance_efficiency], axis=1)
y = np.stack([energy_use, water_use, carbon_emissions], axis=1)
 
# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
 
# Build multi-output regression model
model = models.Sequential([
    layers.Input(shape=(5,)),
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(3)  # Outputs: energy, water, emissions
])
 
model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=0)
 
# Evaluate model
loss, mae = model.evaluate(X_test, y_test)
print(f"✅ Green Building Model MAE (kWh, L, CO₂e): {mae}")
 
# Predict for 5 buildings
preds = model.predict(X_test[:5])
for i in range(5):
    print(f"\nBuilding {i+1}:")
    print(f"  ⚡ Energy Use: {preds[i][0]:.0f} kWh")
    print(f"  💧 Water Use: {preds[i][1]:.0f} L")
    print(f"  🌍 CO₂ Emissions: {preds[i][2]:.0f} kg CO₂e")