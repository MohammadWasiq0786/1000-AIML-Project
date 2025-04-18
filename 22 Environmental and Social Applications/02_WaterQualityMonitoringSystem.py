"""
Project 842: Water Quality Monitoring System
Description
Water quality monitoring is essential for public health, agriculture, and environmental safety. AI can help analyze sensor data to assess whether water is safe or contaminated. In this project, we simulate water parameters (pH, turbidity, dissolved oxygen, etc.) and build a binary classifier to predict water suitability for use.

This model can be extended with:

Real sensor data from field devices (e.g., IoT water sensors)

Integrated into rural water safety dashboards or early alert systems
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
 
# Simulate water quality parameters
np.random.seed(42)
n_samples = 1000
 
pH = np.random.normal(7.0, 0.5, n_samples)                     # Ideal range: 6.5–8.5
turbidity = np.random.normal(3, 1.5, n_samples)                # NTU: lower is better
dissolved_oxygen = np.random.normal(7.5, 1.5, n_samples)       # mg/L
nitrate = np.random.normal(5, 2, n_samples)                    # mg/L
temperature = np.random.normal(25, 3, n_samples)               # °C
 
# Label as suitable (1) or unsuitable (0) for drinking
# Simple rule: too high nitrate, low oxygen, or high turbidity = unsafe
suitable = ((pH >= 6.5) & (pH <= 8.5) &
            (turbidity < 5) &
            (dissolved_oxygen > 5) &
            (nitrate < 10)).astype(int)
 
# Features and labels
X = np.stack([pH, turbidity, dissolved_oxygen, nitrate, temperature], axis=1)
y = suitable
 
# Split into train/test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
 
# Build binary classification model
model = models.Sequential([
    layers.Input(shape=(5,)),
    layers.Dense(32, activation='relu'),
    layers.Dense(16, activation='relu'),
    layers.Dense(1, activation='sigmoid')  # Output: 1 = suitable, 0 = not suitable
])
 
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=15, batch_size=32, verbose=0)
 
# Evaluate
loss, acc = model.evaluate(X_test, y_test)
print(f"✅ Water Quality Model Accuracy: {acc:.4f}")
 
# Predict for a few samples
preds = (model.predict(X_test[:5]) > 0.5).astype(int).flatten()
for i in range(5):
    print(f"Sample {i+1}: {'✅ Suitable' if preds[i] else '❌ Unsuitable'} (Actual: {'✅' if y_test[i] else '❌'})")