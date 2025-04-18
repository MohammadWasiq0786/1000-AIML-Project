"""
Project 864: Carbon Footprint Calculation
Description
Carbon footprint calculators help estimate CO₂ emissions from daily activities—like transportation, energy usage, and diet. In this project, we simulate lifestyle data and build a regression model to estimate a user’s monthly carbon footprint in kilograms of CO₂ equivalent (kg CO₂e).

✅ This model can power:

Carbon tracking apps

Sustainability reports for individuals or companies

Interactive “green living” dashboards
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
 
# Simulate user lifestyle inputs
np.random.seed(42)
n_samples = 1000
 
miles_driven = np.random.normal(800, 200, n_samples)            # km/month
electricity_usage = np.random.normal(300, 100, n_samples)       # kWh/month
meat_consumption = np.random.normal(15, 5, n_samples)           # kg/month
flights = np.random.poisson(0.5, n_samples)                     # flights/month
recycling_rate = np.random.uniform(0, 1, n_samples)             # 0–1 scale
 
# Simulated carbon footprint in kg CO₂e (simple formula with noise)
carbon_footprint = (
    0.2 * miles_driven +
    0.5 * electricity_usage +
    27 * meat_consumption +
    250 * flights -
    100 * recycling_rate +
    np.random.normal(0, 50, n_samples)
)
 
# Combine features and targets
X = np.stack([miles_driven, electricity_usage, meat_consumption, flights, recycling_rate], axis=1)
y = carbon_footprint
 
# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
 
# Build regression model
model = models.Sequential([
    layers.Input(shape=(5,)),
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(1)  # Output: carbon footprint in kg CO₂e
])
 
model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=0)
 
# Evaluate model
loss, mae = model.evaluate(X_test, y_test)
print(f"✅ Carbon Footprint Estimation MAE: {mae:.2f} kg CO₂e")
 
# Predict for 5 users
preds = model.predict(X_test[:5]).flatten()
for i in range(5):
    print(f"User {i+1}: Predicted Carbon Footprint = {preds[i]:.1f} kg CO₂e (Actual: {y_test[i]:.1f})")