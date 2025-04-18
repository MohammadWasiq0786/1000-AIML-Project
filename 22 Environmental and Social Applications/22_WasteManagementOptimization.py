"""
Project 862: Waste Management Optimization
Description
Optimizing waste collection routes and bin servicing helps reduce fuel costs, emissions, and overflow incidents. In this project, we simulate data from smart waste bins and build a regression model to predict optimal pickup priority, helping schedule efficient waste collection.

✅ This model can be integrated with:

Smart bin networks in cities

Routing algorithms for waste trucks

Municipal dashboards for sustainability planning
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
 
# Simulate smart bin sensor data
np.random.seed(42)
n_samples = 1000
 
fill_level = np.random.uniform(0, 1, n_samples)             # 0 (empty) to 1 (full)
last_collected_days = np.random.randint(1, 10, n_samples)   # days
bin_location_density = np.random.normal(50, 20, n_samples)  # people/km²
organic_ratio = np.random.uniform(0, 1, n_samples)          # % organic waste
temperature = np.random.normal(30, 5, n_samples)            # °C (affects odor)
 
# Target: pickup priority score (0 to 1) based on fill, time, population, and organic % in heat
priority_score = (
    0.4 * fill_level +
    0.3 * (last_collected_days / 10) +
    0.1 * (bin_location_density / 100) +
    0.1 * organic_ratio * (temperature / 40) +
    np.random.normal(0, 0.05, n_samples)
)
 
priority_score = np.clip(priority_score, 0, 1)
 
# Combine features
X = np.stack([fill_level, last_collected_days, bin_location_density, organic_ratio, temperature], axis=1)
y = priority_score
 
# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
 
# Build regression model
model = models.Sequential([
    layers.Input(shape=(5,)),
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(1)  # Predict pickup priority (0–1)
])
 
model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=0)
 
# Evaluate model
loss, mae = model.evaluate(X_test, y_test)
print(f"✅ Waste Pickup Priority Prediction MAE: {mae:.2f}")
 
# Predict for a few bins
preds = model.predict(X_test[:5]).flatten()
for i in range(5):
    print(f"Bin {i+1}: Pickup Priority Score = {preds[i]:.2f} (Actual: {y_test[i]:.2f})")