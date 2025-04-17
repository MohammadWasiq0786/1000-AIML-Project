"""
Project 787: Smart Retail Analytics
Description
Smart retail analytics uses sensors and AI to understand customer behavior, optimize store layouts, manage inventory, and personalize marketing. In this project, we simulate footfall, dwell time, and product interaction data to predict purchase likelihood — enabling dynamic promotions and better customer targeting.

This model supports smart shelf systems, in-store analytics dashboards, or dynamic signage — and can be connected to edge devices collecting sensor data (e.g., footfall counters, RFID readers).
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
 
# Simulate retail features: time in store (mins), sections visited, items picked up, interaction time (secs)
np.random.seed(42)
n_samples = 1000
 
time_in_store = np.random.normal(15, 5, n_samples)
sections_visited = np.random.randint(1, 10, n_samples)
items_touched = np.random.randint(0, 5, n_samples)
interaction_time = np.random.normal(30, 15, n_samples)
 
# Label: 1 = purchase likely if more interactions and longer time spent
purchase = ((time_in_store > 12) & (items_touched >= 2) & (interaction_time > 25)).astype(int)
 
# Combine features
X = np.stack([time_in_store, sections_visited, items_touched, interaction_time], axis=1)
y = purchase
 
# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
 
# Build binary classifier model
model = models.Sequential([
    layers.Input(shape=(4,)),
    layers.Dense(32, activation='relu'),
    layers.Dense(16, activation='relu'),
    layers.Dense(1, activation='sigmoid')  # Binary output: will purchase or not
])
 
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=15, batch_size=32, verbose=0)
 
# Evaluate model
loss, acc = model.evaluate(X_test, y_test)
print(f"✅ Smart Retail Purchase Prediction Accuracy: {acc:.4f}")
 
# Predict behavior for 5 customers
preds = (model.predict(X_test[:5]) > 0.5).astype(int).flatten()
for i in range(5):
    print(f"Customer {i+1}: {'Will Purchase' if preds[i] else 'No Purchase'} (Actual: {'Will Purchase' if y_test[i] else 'No Purchase'})")