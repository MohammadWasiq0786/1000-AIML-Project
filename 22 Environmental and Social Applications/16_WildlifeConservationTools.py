"""
Project 856: Wildlife Conservation Tools
Description
Wildlife conservation tools leverage AI to track endangered species, detect poaching risks, and allocate ranger patrols. In this project, we simulate ecological, human activity, and terrain data to build a binary classification model that predicts conservation threat level (Low or High) in a region — helping prioritize monitoring and intervention.

✅ This model supports:

Ranger patrol route planning

Automated alerts in protected zones

Integration with satellite imagery and geofencing tools
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
 
# Simulate ecological data
np.random.seed(42)
n_samples = 1000
 
animal_density = np.random.normal(20, 5, n_samples)         # animals/km²
human_activity = np.random.normal(30, 10, n_samples)        # vehicles/day or movement events
vegetation_cover = np.random.normal(0.7, 0.1, n_samples)    # NDVI index
distance_to_road = np.random.normal(15, 7, n_samples)       # km
protection_score = np.random.normal(0.6, 0.2, n_samples)    # 0 = no protection, 1 = high
 
# Label: High risk if human activity is high, animal density is high, and protection is low
threat_level = (
    (animal_density > 22) &
    (human_activity > 35) &
    (protection_score < 0.5)
).astype(int)
 
# Stack features
X = np.stack([animal_density, human_activity, vegetation_cover, distance_to_road, protection_score], axis=1)
y = threat_level
 
# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
 
# Build binary classifier model
model = models.Sequential([
    layers.Input(shape=(5,)),
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(1, activation='sigmoid')  # Output: 1 = High threat
])
 
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=15, batch_size=32, verbose=0)
 
# Evaluate model
loss, acc = model.evaluate(X_test, y_test)
print(f"✅ Wildlife Threat Assessment Accuracy: {acc:.4f}")
 
# Predict for 5 regions
preds = (model.predict(X_test[:5]) > 0.5).astype(int).flatten()
for i in range(5):
    print(f"Region {i+1}: {'🚨 High Risk' if preds[i] else '✅ Low Risk'} (Actual: {'🚨' if y_test[i] else '✅'})")