"""
Project 868: Urban Planning Assistance
Description
Urban planning assistance tools help cities make data-driven decisions on zoning, infrastructure, and amenities. In this project, we simulate demographic and spatial features to build a multi-output regression model that predicts population density, infrastructure need, and green space requirement in city zones.

✅ This tool can support:

City master plans

Data-driven zoning proposals

Green infrastructure and sustainability mapping
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
 
# Simulate urban zone data
np.random.seed(42)
n_samples = 1000
 
avg_income = np.random.normal(30000, 8000, n_samples)          # USD
distance_to_center = np.random.normal(8, 4, n_samples)         # km
current_density = np.random.normal(1000, 300, n_samples)       # people/km²
public_transport_score = np.random.uniform(0, 1, n_samples)    # 0–1
school_access_score = np.random.uniform(0, 1, n_samples)       # 0–1
 
# Simulated urban outputs:
# 1. Projected population density (people/km²)
# 2. Infrastructure need index (0–100)
# 3. Recommended green space (hectares)
pop_density = current_density + np.random.normal(50, 30, n_samples)
infra_need = (1 - public_transport_score) * 50 + (1 - school_access_score) * 50 + np.random.normal(0, 5, n_samples)
green_space = np.maximum(10 - (current_density / 200) + (distance_to_center / 2), 0) + np.random.normal(0, 1, n_samples)
 
# Stack features and labels
X = np.stack([avg_income, distance_to_center, current_density, public_transport_score, school_access_score], axis=1)
y = np.stack([pop_density, infra_need, green_space], axis=1)
 
# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
 
# Build multi-output regression model
model = models.Sequential([
    layers.Input(shape=(5,)),
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(3)  # 3 outputs: population, infra, green space
])
 
model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=0)
 
# Evaluate model
loss, mae = model.evaluate(X_test, y_test)
print(f"✅ Urban Planning Model MAE (Pop, Infra, Green): {mae}")
 
# Predict for 5 urban zones
preds = model.predict(X_test[:5])
for i in range(5):
    print(f"\nZone {i+1} Predictions:")
    print(f"  🏙️ Projected Density: {preds[i][0]:.1f} ppl/km²")
    print(f"  🏗️ Infrastructure Need: {preds[i][1]:.1f}")
    print(f"  🌳 Green Space Required: {preds[i][2]:.1f} ha")