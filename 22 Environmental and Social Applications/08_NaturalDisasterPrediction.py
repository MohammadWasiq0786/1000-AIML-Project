"""
Project 848: Natural Disaster Prediction
Description
Predicting natural disasters like earthquakes, hurricanes, or landslides helps save lives and reduce damage. In this project, we simulate environmental and geophysical features and build a binary classification model to predict the likelihood of a natural disaster event (yes/no), based on regional sensor data.

✅ Real-world use cases:

Disaster early warning systems

Emergency response automation

Smart cities & climate-resilient infrastructure planning
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
 
# Simulate sensor-based environmental data
np.random.seed(42)
n_samples = 1000
 
seismic_activity = np.random.normal(3.0, 1.0, n_samples)      # Richter scale
rainfall = np.random.normal(100, 50, n_samples)               # mm
wind_speed = np.random.normal(40, 20, n_samples)              # km/h
soil_saturation = np.random.uniform(0, 1, n_samples)          # 0 to 1
temperature = np.random.normal(30, 5, n_samples)              # °C
 
# Label: 1 = disaster if high seismic + heavy rain + soil saturation or strong winds
disaster = ((seismic_activity > 5) | 
            ((rainfall > 150) & (soil_saturation > 0.8)) | 
            (wind_speed > 80)).astype(int)
 
# Feature matrix and labels
X = np.stack([seismic_activity, rainfall, wind_speed, soil_saturation, temperature], axis=1)
y = disaster
 
# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
 
# Build binary classifier model
model = models.Sequential([
    layers.Input(shape=(5,)),
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(1, activation='sigmoid')  # Output: 1 = likely disaster
])
 
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=15, batch_size=32, verbose=0)
 
# Evaluate
loss, acc = model.evaluate(X_test, y_test)
print(f"✅ Natural Disaster Prediction Accuracy: {acc:.4f}")
 
# Predict and display results for 5 locations
preds = (model.predict(X_test[:5]) > 0.5).astype(int).flatten()
for i in range(5):
    print(f"Location {i+1}: {'⚠️ Disaster Likely' if preds[i] else '✅ Normal'} (Actual: {'⚠️' if y_test[i] else '✅'})")