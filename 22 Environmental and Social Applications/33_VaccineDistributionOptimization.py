"""
Project 873: Vaccine Distribution Optimization
Description
Efficient vaccine distribution is crucial during pandemics or routine immunization campaigns. In this project, we simulate health, logistics, and demographic data and build a regression model to predict the optimal daily vaccine allocation per region, balancing demand, risk, and cold-chain capacity.

✅ Practical uses:

Vaccine logistics planning

Cold-chain optimization

Emergency health response simulation tools
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
 
# Simulate region-level data
np.random.seed(42)
n_samples = 1000
 
population = np.random.normal(100000, 20000, n_samples)            # people
infection_rate = np.random.normal(0.02, 0.01, n_samples)           # cases per capita
elderly_ratio = np.random.normal(0.15, 0.05, n_samples)            # % elderly
logistics_score = np.random.uniform(0, 1, n_samples)               # 0–1 (1 = excellent)
cold_chain_capacity = np.random.normal(5000, 1000, n_samples)      # vaccines/day
 
# Target: vaccine allocation needed per day
vaccine_demand = (
    population * infection_rate * 0.4 +                     # 40% target coverage of infected
    elderly_ratio * population * 0.3 +                      # prioritize elderly
    cold_chain_capacity * 0.6 +                             # logistics-capable delivery
    np.random.normal(0, 500, n_samples)                     # some randomness
)
 
# Feature matrix and labels
X = np.stack([
    population,
    infection_rate,
    elderly_ratio,
    logistics_score,
    cold_chain_capacity
], axis=1)
y = vaccine_demand
 
# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
 
# Build regression model
model = models.Sequential([
    layers.Input(shape=(5,)),
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(1)  # Output: vaccine allocation per day
])
 
model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=0)
 
# Evaluate performance
loss, mae = model.evaluate(X_test, y_test)
print(f"✅ Vaccine Distribution Model MAE: {mae:.0f} doses/day")
 
# Predict for 5 regions
preds = model.predict(X_test[:5]).flatten()
for i in range(5):
    print(f"Region {i+1}: Predicted Allocation = {preds[i]:,.0f} doses/day (Actual: {y_test[i]:,.0f})")