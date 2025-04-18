"""
Project 843: Climate Change Impact Assessment
Description
Climate change impacts various environmental and socio-economic factors, including temperature, precipitation, crop yield, and sea level rise. In this project, we simulate climate indicators over time and build a multi-output regression model to assess potential impacts such as temperature anomaly and crop productivity loss, based on rising greenhouse gas (GHG) levels and other environmental indicators.

This model can be:

Expanded with real-world climate data (e.g., from NASA, IPCC, NOAA)

Adapted for geospatial visualization, policy simulation, or agricultural planning
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
 
# Simulate global climate indicators
np.random.seed(42)
n_samples = 1000
 
# Features: CO2 (ppm), Methane (CH4 in ppb), Nitrous Oxide (N2O in ppb), Deforestation rate (%), Sea ice loss (%)
co2 = np.random.normal(420, 20, n_samples)
ch4 = np.random.normal(1900, 150, n_samples)
n2o = np.random.normal(330, 15, n_samples)
deforestation = np.random.normal(1.5, 0.5, n_samples)
sea_ice_loss = np.random.normal(3, 1, n_samples)
 
# Outputs: temperature anomaly (°C), crop productivity loss (%)
temp_anomaly = 0.01 * co2 + 0.005 * ch4 + 0.008 * deforestation + np.random.normal(0, 0.2, n_samples)
crop_loss = 0.03 * deforestation + 0.02 * temp_anomaly + 0.01 * sea_ice_loss + np.random.normal(0, 1, n_samples)
 
# Input features and output targets
X = np.stack([co2, ch4, n2o, deforestation, sea_ice_loss], axis=1)
y = np.stack([temp_anomaly, crop_loss], axis=1)
 
# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
 
# Build multi-output regression model
model = models.Sequential([
    layers.Input(shape=(5,)),
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(2)  # 2 outputs: temperature anomaly and crop loss
])
 
model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=0)
 
# Evaluate model
loss, mae = model.evaluate(X_test, y_test)
print(f"✅ Climate Impact Model MAE (Temp Anomaly, Crop Loss): {mae}")
 
# Predict for first 5 samples
preds = model.predict(X_test[:5])
for i in range(5):
    print(f"Sample {i+1}: Predicted Temp Anomaly = {preds[i][0]:.2f}°C, Predicted Crop Loss = {preds[i][1]:.2f}%")