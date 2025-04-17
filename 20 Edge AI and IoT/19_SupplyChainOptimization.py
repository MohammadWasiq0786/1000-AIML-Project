"""
Project 779: Supply Chain Optimization
Description
Supply chain optimization involves predicting demand, managing inventory, and scheduling shipments to reduce costs and delays. AI helps by forecasting product demand and optimizing stock levels. In this simulation, we'll use historical order data to predict future demand using a regression model (DNN), which can be used to automate restocking decisions.

This model helps businesses anticipate inventory needs, reduce waste, and automate restocking. It can be deployed on cloud dashboards or edge-based warehouse systems using lightweight AI inference.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
 
# Simulate historical demand features for products
np.random.seed(42)
n_samples = 1000
 
# Features: past week demand, days since last restock, current stock level, promo flag
past_week_demand = np.random.randint(10, 200, n_samples)
days_since_restock = np.random.randint(1, 30, n_samples)
current_stock = np.random.randint(0, 300, n_samples)
promotion = np.random.choice([0, 1], n_samples)
 
# Target: next week's demand (somewhat correlated with past demand and promotion)
next_week_demand = past_week_demand * (1.1 + 0.2 * promotion) + np.random.normal(0, 10, n_samples)
 
# Feature matrix and target
X = np.stack([past_week_demand, days_since_restock, current_stock, promotion], axis=1)
y = next_week_demand
 
# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
 
# Build regression model
model = models.Sequential([
    layers.Input(shape=(4,)),
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(1)  # Predicted demand
])
 
model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=0)
 
# Evaluate performance
loss, mae = model.evaluate(X_test, y_test)
print(f"✅ Supply Chain Demand Forecasting MAE: {mae:.2f} units")
 
# Predict demand for next 5 entries
predictions = model.predict(X_test[:5]).flatten()
for i, pred in enumerate(predictions):
    print(f"Sample {i+1}: Predicted Demand = {pred:.1f} units (Actual = {y_test[i]:.1f})")