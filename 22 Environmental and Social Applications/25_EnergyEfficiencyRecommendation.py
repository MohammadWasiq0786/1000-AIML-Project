"""
Project 865: Energy Efficiency Recommendation System
Description
This system analyzes household energy usage patterns and recommends ways to save electricity, reduce bills, and lower environmental impact. In this project, we simulate energy consumption data and build a multi-class classification model to recommend an energy-saving action (e.g., switch to LED, improve insulation, upgrade appliances, etc.).

✅ Great for:

Home energy audit apps

Smart home dashboards

Sustainability-focused utility tools
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
 
# Simulate household data
np.random.seed(42)
n_samples = 1000
 
monthly_kwh = np.random.normal(400, 100, n_samples)          # energy usage in kWh/month
avg_temp = np.random.normal(22, 5, n_samples)                # average indoor temperature
appliance_age = np.random.normal(8, 3, n_samples)            # years
lighting_ratio = np.random.uniform(0, 1, n_samples)          # % LED bulbs
insulation_score = np.random.uniform(0, 1, n_samples)        # 0 (poor) to 1 (excellent)
 
# Recommendations:
# 0 = Switch to LED
# 1 = Improve Insulation
# 2 = Upgrade Appliances
# 3 = Use Smart Thermostat
 
recommendation = np.where(
    lighting_ratio < 0.3, 0,
    np.where(insulation_score < 0.4, 1,
    np.where(appliance_age > 10, 2, 3))
)
 
# Stack features
X = np.stack([monthly_kwh, avg_temp, appliance_age, lighting_ratio, insulation_score], axis=1)
y = recommendation
 
# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
 
# Build multi-class classifier
model = models.Sequential([
    layers.Input(shape=(5,)),
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(4, activation='softmax')  # 4 recommendations
])
 
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=15, batch_size=32, verbose=0)
 
# Evaluate performance
loss, acc = model.evaluate(X_test, y_test)
print(f"✅ Energy Efficiency Recommender Accuracy: {acc:.4f}")
 
# Predict for 5 households
preds = np.argmax(model.predict(X_test[:5]), axis=1)
rec_map = {
    0: "💡 Switch to LED",
    1: "🏠 Improve Insulation",
    2: "🔌 Upgrade Appliances",
    3: "🌡️ Use Smart Thermostat"
}
 
for i in range(5):
    print(f"Home {i+1}: Recommended = {rec_map[preds[i]]}, Actual = {rec_map[y_test[i]]}")