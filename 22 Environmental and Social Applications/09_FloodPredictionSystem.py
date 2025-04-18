"""
Project 849: Flood Prediction System
Description
Flood prediction systems help anticipate flash floods and river overflows, enabling timely evacuation and resource allocation. In this project, we simulate hydrological and meteorological data (e.g., rainfall, river level, soil moisture) and build a binary classifier to predict the likelihood of flooding in a given region.

✅ This type of system supports:

Real-time flood alerts from IoT water level sensors

Disaster preparedness tools

Integration with GIS dashboards and rescue team dispatch systems
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
 
# Simulate relevant flood predictors
np.random.seed(42)
n_samples = 1000
 
rainfall = np.random.normal(120, 40, n_samples)              # mm
river_level = np.random.normal(3.5, 1.0, n_samples)          # meters
soil_moisture = np.random.uniform(0, 1, n_samples)           # 0 = dry, 1 = saturated
runoff_rate = np.random.normal(50, 15, n_samples)            # mm/hr
catchment_slope = np.random.normal(10, 5, n_samples)         # degrees
 
# Label: 1 = flood likely if high rain + high river + saturated soil or high runoff
flood = ((rainfall > 150) & (river_level > 4) & (soil_moisture > 0.8) |
         (runoff_rate > 70)).astype(int)
 
# Combine features
X = np.stack([rainfall, river_level, soil_moisture, runoff_rate, catchment_slope], axis=1)
y = flood
 
# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
 
# Build binary classifier
model = models.Sequential([
    layers.Input(shape=(5,)),
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(1, activation='sigmoid')  # Output: 1 = flood risk
])
 
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=15, batch_size=32, verbose=0)
 
# Evaluate
loss, acc = model.evaluate(X_test, y_test)
print(f"✅ Flood Prediction Model Accuracy: {acc:.4f}")
 
# Predict risk for first 5 locations
preds = (model.predict(X_test[:5]) > 0.5).astype(int).flatten()
for i in range(5):
    print(f"Region {i+1}: {'🌊 Flood Likely' if preds[i] else '✅ Safe'} (Actual: {'🌊' if y_test[i] else '✅'})")