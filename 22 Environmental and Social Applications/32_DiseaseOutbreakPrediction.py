"""
Project 872: Disease Outbreak Prediction
Description
Early outbreak prediction helps public health officials respond before widespread transmission occurs. In this project, we simulate environmental, mobility, and health indicators to build a binary classification model that predicts the likelihood of a disease outbreak in a given region and time window.

✅ Applications include:

Early warning dashboards

NGO/public health planning

Epidemic modeling & simulation platforms
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
 
# Simulate region-wise features
np.random.seed(42)
n_samples = 1000
 
recent_case_rate = np.random.normal(20, 10, n_samples)         # cases per 10,000 people
population_density = np.random.normal(3000, 1000, n_samples)   # people/km²
mobility_index = np.random.normal(1.2, 0.4, n_samples)         # avg contacts/person/day
vaccination_rate = np.random.uniform(0, 1, n_samples)          # 0–1 scale
temp_anomaly = np.random.normal(0.5, 0.3, n_samples)           # °C deviation from normal
 
# Label: 1 = outbreak if high case rate + mobility + low vax + high density or temperature anomaly
outbreak = (
    (recent_case_rate > 25) &
    (mobility_index > 1.5) &
    (vaccination_rate < 0.4) &
    ((population_density > 3500) | (temp_anomaly > 0.6))
).astype(int)
 
# Stack features
X = np.stack([recent_case_rate, population_density, mobility_index, vaccination_rate, temp_anomaly], axis=1)
y = outbreak
 
# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
 
# Build binary classifier
model = models.Sequential([
    layers.Input(shape=(5,)),
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(1, activation='sigmoid')  # Output: outbreak risk
])
 
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=0)
 
# Evaluate
loss, acc = model.evaluate(X_test, y_test)
print(f"✅ Disease Outbreak Prediction Accuracy: {acc:.4f}")
 
# Predict risk for 5 regions
preds = (model.predict(X_test[:5]) > 0.5).astype(int).flatten()
for i in range(5):
    print(f"Region {i+1}: {'🚨 Outbreak Likely' if preds[i] else '✅ No Outbreak'} (Actual: {'🚨' if y_test[i] else '✅'})")