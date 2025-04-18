"""
Project 850: Wildfire Risk Assessment
Description
Wildfire risk assessment predicts the likelihood of fires based on weather, vegetation, and land conditions. This helps governments and forest managers deploy resources and issue early warnings. In this project, we simulate key environmental factors and build a binary classification model to predict wildfire risk (yes/no).

✅ Real-world applications include:

Forest fire early warning systems

Satellite + sensor data fusion

Integration with drones or fire watch towers for real-time decision-making
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
 
# Simulate environmental features
np.random.seed(42)
n_samples = 1000
 
temperature = np.random.normal(35, 5, n_samples)          # °C
humidity = np.random.normal(30, 10, n_samples)            # %
wind_speed = np.random.normal(20, 5, n_samples)           # km/h
vegetation_dryness = np.random.uniform(0, 1, n_samples)   # 0 = wet, 1 = dry
recent_rain = np.random.normal(5, 3, n_samples)           # mm in last 7 days
 
# Label: wildfire likely if high temp + low humidity + dryness + low rainfall + high wind
fire_risk = (
    (temperature > 35) &
    (humidity < 25) &
    (vegetation_dryness > 0.7) &
    (recent_rain < 5) &
    (wind_speed > 20)
).astype(int)
 
# Stack features and labels
X = np.stack([temperature, humidity, wind_speed, vegetation_dryness, recent_rain], axis=1)
y = fire_risk
 
# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
 
# Build binary classification model
model = models.Sequential([
    layers.Input(shape=(5,)),
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(1, activation='sigmoid')  # Output: fire risk (yes/no)
])
 
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=15, batch_size=32, verbose=0)
 
# Evaluate model
loss, acc = model.evaluate(X_test, y_test)
print(f"✅ Wildfire Risk Model Accuracy: {acc:.4f}")
 
# Predict for sample areas
preds = (model.predict(X_test[:5]) > 0.5).astype(int).flatten()
for i in range(5):
    print(f"Zone {i+1}: {'🔥 High Risk' if preds[i] else '✅ Low Risk'} (Actual: {'🔥' if y_test[i] else '✅'})")