"""
Project 870: Water Resource Management
Description
Water resource management systems forecast demand and monitor usage to ensure efficient distribution across urban, agricultural, and industrial sectors. In this project, we simulate sector-wise water usage data and build a multi-output regression model to predict future water demand for domestic, agricultural, and industrial use.

✅ This model is useful for:

Water utilities and supply networks

Agricultural planning tools

Drought resilience dashboards
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
 
# Simulate region-level resource data
np.random.seed(42)
n_samples = 1000
 
population = np.random.normal(100000, 20000, n_samples)         # people
avg_temp = np.random.normal(30, 5, n_samples)                   # °C
crop_area = np.random.normal(1500, 500, n_samples)              # hectares
industry_units = np.random.normal(200, 50, n_samples)           # factories/plants
precipitation = np.random.normal(80, 20, n_samples)             # mm/month
 
# Targets: Water demand by sector (liters/day)
domestic_use = population * 150 + np.random.normal(0, 50000, n_samples)       # 150 L/person/day
agricultural_use = crop_area * 5000 - precipitation * crop_area * 10          # adjust by rain
industrial_use = industry_units * 20000 + avg_temp * 500
 
# Feature matrix and targets
X = np.stack([population, avg_temp, crop_area, industry_units, precipitation], axis=1)
y = np.stack([domestic_use, agricultural_use, industrial_use], axis=1)
 
# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
 
# Build multi-output regression model
model = models.Sequential([
    layers.Input(shape=(5,)),
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(3)  # Output: domestic, agri, industrial
])
 
model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=0)
 
# Evaluate model
loss, mae = model.evaluate(X_test, y_test)
print(f"✅ Water Demand Forecast MAE: {mae}")
 
# Predict for 5 regions
preds = model.predict(X_test[:5])
for i in range(5):
    print(f"\nRegion {i+1} Water Demand Forecast:")
    print(f"  🏠 Domestic: {preds[i][0]:,.0f} L/day")
    print(f"  🌾 Agricultural: {preds[i][1]:,.0f} L/day")
    print(f"  🏭 Industrial: {preds[i][2]:,.0f} L/day")