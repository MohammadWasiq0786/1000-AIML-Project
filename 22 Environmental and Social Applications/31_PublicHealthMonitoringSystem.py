"""
Project 871: Public Health Monitoring System
Description
A public health monitoring system detects trends in disease symptoms, environmental risks, and healthcare usage to support early response and resource planning. In this project, we simulate demographic and health signals and build a multi-output classification model to predict health risk categories (e.g., respiratory, waterborne, or chronic risks) across different zones.

✅ This model supports:

City-level health dashboards

Mobile disease surveillance

Proactive public health response systems
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
 
# Simulate region-wise health and environment data
np.random.seed(42)
n_samples = 1000
 
air_quality_index = np.random.normal(120, 40, n_samples)       # AQI
water_contamination_index = np.random.normal(0.3, 0.2, n_samples)  # 0–1 scale
population_density = np.random.normal(3000, 1000, n_samples)   # people/km²
elderly_ratio = np.random.normal(0.12, 0.05, n_samples)        # % elderly
healthcare_access_score = np.random.uniform(0, 1, n_samples)   # 0–1
 
# Multi-label targets:
# - Respiratory risk (high AQI + elderly + dense pop)
# - Waterborne disease risk (contaminated water + low healthcare)
# - Chronic disease risk (high elderly + low access)
 
resp_risk = ((air_quality_index > 150) & (elderly_ratio > 0.1) & (population_density > 3500)).astype(int)
water_risk = ((water_contamination_index > 0.4) & (healthcare_access_score < 0.5)).astype(int)
chronic_risk = ((elderly_ratio > 0.15) & (healthcare_access_score < 0.4)).astype(int)
 
# Features and labels
X = np.stack([air_quality_index, water_contamination_index, population_density, elderly_ratio, healthcare_access_score], axis=1)
y = np.stack([resp_risk, water_risk, chronic_risk], axis=1)
 
# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
 
# Build multi-label classification model
model = models.Sequential([
    layers.Input(shape=(5,)),
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(3, activation='sigmoid')  # 3 health risk outputs
])
 
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=15, batch_size=32, verbose=0)
 
# Evaluate model
loss, acc = model.evaluate(X_test, y_test)
print(f"✅ Public Health Monitoring Accuracy: {acc:.4f}")
 
# Predict risk for 5 regions
preds = (model.predict(X_test[:5]) > 0.5).astype(int)
risk_labels = ['🫁 Respiratory', '💧 Waterborne', '❤️ Chronic']
 
for i in range(5):
    print(f"\nRegion {i+1} Risks:")
    for j in range(3):
        print(f"  {risk_labels[j]}: {'⚠️ Risk' if preds[i][j] else '✅ Safe'} (Actual: {'⚠️' if y_test[i][j] else '✅'})")