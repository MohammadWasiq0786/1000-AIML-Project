"""
Project 866: Sustainable Transportation Planning
Description
Sustainable transportation planning involves analyzing commuting patterns, emissions, and access to public transport to promote greener alternatives. In this project, we simulate transportation data and build a multi-class classification model to recommend a sustainable commute option (e.g., Bike, Bus, EV Car, Walk).

✅ Useful for:

Urban mobility planning

Green commuting apps

Sustainable workplace travel dashboards
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
 
# Simulate commuter profiles
np.random.seed(42)
n_samples = 1000
 
commute_distance = np.random.normal(10, 5, n_samples)        # km
emissions_per_km = np.random.normal(0.15, 0.05, n_samples)   # kg CO2/km
public_transport_access = np.random.uniform(0, 1, n_samples) # 0–1 scale
bike_lane_score = np.random.uniform(0, 1, n_samples)         # 0–1 scale
walking_feasibility = np.random.uniform(0, 1, n_samples)     # 0–1 scale
 
# Recommendations:
# 0 = 🚲 Bike
# 1 = 🚌 Public Transit
# 2 = 🚗 Switch to EV
# 3 = 🚶 Walk
 
recommendation = np.where(
    (commute_distance < 5) & (walking_feasibility > 0.7), 3,
    np.where((commute_distance < 8) & (bike_lane_score > 0.6), 0,
    np.where((public_transport_access > 0.5), 1, 2))
)
 
# Feature matrix
X = np.stack([
    commute_distance,
    emissions_per_km,
    public_transport_access,
    bike_lane_score,
    walking_feasibility
], axis=1)
 
y = recommendation
 
# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
 
# Build model
model = models.Sequential([
    layers.Input(shape=(5,)),
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(4, activation='softmax')  # 4 commute recommendations
])
 
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=15, batch_size=32, verbose=0)
 
# Evaluate
loss, acc = model.evaluate(X_test, y_test)
print(f"✅ Sustainable Commute Recommender Accuracy: {acc:.4f}")
 
# Predict commute suggestions
preds = np.argmax(model.predict(X_test[:5]), axis=1)
rec_map = {
    0: "🚲 Bike",
    1: "🚌 Public Transit",
    2: "🚗 Switch to EV",
    3: "🚶 Walk"
}
 
for i in range(5):
    print(f"Commuter {i+1}: Suggested = {rec_map[preds[i]]}, Actual = {rec_map[y_test[i]]}")