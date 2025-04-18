"""
Project 859: Deforestation Monitoring
Description
Deforestation monitoring helps track forest loss over time using satellite imagery and vegetation indices. It enables early intervention to combat illegal logging and protect biodiversity. In this project, we simulate remote sensing data across time and build a binary classifier to detect deforestation events in a given region.

✅ This model can be extended with:

NDVI time-series from Landsat/Sentinel satellites

Geo-tagged logging activity

Integrated into real-time deforestation alert platforms like Global Forest Watch
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
 
# Simulate satellite data for forest regions
np.random.seed(42)
n_samples = 1000
 
# Features: NDVI drop, surface temp rise, logging activity, time since last rain, proximity to roads
ndvi_change = np.random.normal(-0.1, 0.2, n_samples)           # NDVI drop over time
surface_temp_rise = np.random.normal(1.5, 0.5, n_samples)      # °C rise
logging_index = np.random.normal(0.4, 0.3, n_samples)          # 0–1 (higher = more logging)
days_since_rain = np.random.normal(15, 5, n_samples)           # days
distance_to_road = np.random.normal(5, 2, n_samples)           # km
 
# Label: 1 = deforested area, based on sharp NDVI drop + logging + dryness + road proximity
deforested = (
    (ndvi_change < -0.2) &
    (logging_index > 0.5) &
    (days_since_rain > 10) &
    (distance_to_road < 6)
).astype(int)
 
# Feature matrix and labels
X = np.stack([ndvi_change, surface_temp_rise, logging_index, days_since_rain, distance_to_road], axis=1)
y = deforested
 
# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
 
# Build binary classifier
model = models.Sequential([
    layers.Input(shape=(5,)),
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(1, activation='sigmoid')  # 1 = deforestation
])
 
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=15, batch_size=32, verbose=0)
 
# Evaluate
loss, acc = model.evaluate(X_test, y_test)
print(f"✅ Deforestation Detection Accuracy: {acc:.4f}")
 
# Predict for 5 sample areas
preds = (model.predict(X_test[:5]) > 0.5).astype(int).flatten()
for i in range(5):
    print(f"Area {i+1}: {'🌲 Deforested' if preds[i] else '✅ Forest Intact'} (Actual: {'🌲' if y_test[i] else '✅'})")