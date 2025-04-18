"""
Project 855: Biodiversity Monitoring System
Description
A biodiversity monitoring system helps track species presence, population trends, and ecosystem health using data from camera traps, audio sensors, or environmental DNA. In this project, we simulate ecological observations and build a multi-label classification model to detect the presence of multiple species in a given area.

else '❌'})")
✅ This model can be deployed with:

Acoustic monitoring devices

Camera trap image pipelines

Environmental sensors in conservation areas
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
 
# Simulate environmental observation data (e.g., sound level, vegetation index, time of day, temp)
np.random.seed(42)
n_samples = 1000
 
sound_activity = np.random.normal(50, 15, n_samples)        # dB
vegetation_index = np.random.normal(0.6, 0.1, n_samples)     # NDVI
time_of_day = np.random.randint(0, 24, n_samples)            # hour
temperature = np.random.normal(22, 5, n_samples)             # °C
 
# Simulate presence/absence of 3 species: bird, monkey, insect
# Multi-label (each species is a binary label)
bird_present = ((sound_activity > 40) & (time_of_day < 10)).astype(int)
monkey_present = ((vegetation_index > 0.65) & (time_of_day > 6) & (time_of_day < 18)).astype(int)
insect_present = ((temperature > 20) & (time_of_day >= 18)).astype(int)
 
# Stack features and labels
X = np.stack([sound_activity, vegetation_index, time_of_day, temperature], axis=1)
y = np.stack([bird_present, monkey_present, insect_present], axis=1)
 
# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
 
# Build multi-label classification model
model = models.Sequential([
    layers.Input(shape=(4,)),
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(3, activation='sigmoid')  # 3 species, independent outputs
])
 
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=15, batch_size=32, verbose=0)
 
# Evaluate model
loss, acc = model.evaluate(X_test, y_test)
print(f"✅ Biodiversity Model Accuracy: {acc:.4f}")
 
# Predict species presence for 5 regions
preds = (model.predict(X_test[:5]) > 0.5).astype(int)
species = ['🕊️ Bird', '🐒 Monkey', '🦗 Insect']
for i in range(5):
    print(f"\nRegion {i+1} Species Detected:")
    for j in range(3):
        status = '✅ Present' if preds[i][j] else '❌ Absent'
        print(f"  {species[j]}: {status} (Actual: {'✅' if y_test[i][j] else '❌'})")