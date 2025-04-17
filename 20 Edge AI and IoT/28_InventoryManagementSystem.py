"""
Project 788: Inventory Management System
Description
An inventory management system keeps track of stock levels, predicts shortages, and automates reordering to avoid overstocking or stockouts. This project simulates product usage patterns and builds a classification model to predict whether an item will go “Out of Stock Soon” — ideal for automated restocking alerts in retail or warehouses.

This predictive alert system can integrate with warehouse dashboards, ERP software, or even IoT-connected stock bins for real-time replenishment automation.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
 
# Simulate inventory data: current stock, avg daily sales, days since last restock, incoming shipment (units)
np.random.seed(42)
n_samples = 1000
 
current_stock = np.random.randint(0, 500, n_samples)
daily_sales = np.random.normal(20, 5, n_samples)
days_since_restock = np.random.randint(1, 30, n_samples)
incoming_shipment = np.random.randint(0, 200, n_samples)
 
# Label: 1 = likely out of stock soon if sales high and stock low
out_of_stock_soon = ((current_stock < 100) & (daily_sales > 25) & (incoming_shipment < 50)).astype(int)
 
# Feature matrix and labels
X = np.stack([current_stock, daily_sales, days_since_restock, incoming_shipment], axis=1)
y = out_of_stock_soon
 
# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
 
# Build binary classification model
model = models.Sequential([
    layers.Input(shape=(4,)),
    layers.Dense(32, activation='relu'),
    layers.Dense(16, activation='relu'),
    layers.Dense(1, activation='sigmoid')  # Predict: stock alert (yes/no)
])
 
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=15, batch_size=32, verbose=0)
 
# Evaluate model
loss, acc = model.evaluate(X_test, y_test)
print(f"✅ Inventory Alert Prediction Accuracy: {acc:.4f}")
 
# Predict for 5 example products
preds = (model.predict(X_test[:5]) > 0.5).astype(int).flatten()
for i in range(5):
    print(f"Product {i+1}: {'OUT OF STOCK SOON' if preds[i] else 'Stock OK'} (Actual: {'OUT OF STOCK SOON' if y_test[i] else 'Stock OK'})")