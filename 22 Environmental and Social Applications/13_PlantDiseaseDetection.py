"""
Project 853: Plant Disease Detection
Description
Plant disease detection helps prevent crop loss by identifying issues early. Using AI vision models, we can detect diseases from leaf images. In this project, we simulate a basic image classification pipeline using a CNN model to detect whether a plant leaf is healthy or diseased.

✅ This model can be trained with:

Real leaf datasets like PlantVillage

Mobile-friendly models (e.g., TFLite, MobileNet) for farmers' phones or drones
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
 
# Simulate leaf images as 64x64 grayscale (you can replace with real image dataset like PlantVillage)
np.random.seed(42)
n_samples = 1000
image_size = 64
 
# Healthy leaves: more uniform texture
healthy_images = np.random.normal(loc=0.5, scale=0.1, size=(n_samples // 2, image_size, image_size, 1))
 
# Diseased leaves: add localized noise to simulate spots/patches
diseased_images = healthy_images.copy()
for img in diseased_images:
    x, y = np.random.randint(0, image_size, 10), np.random.randint(0, image_size, 10)
    img[x, y] += np.random.normal(loc=0.3, scale=0.2)
 
# Combine and label
X = np.vstack([healthy_images, diseased_images])
y = np.array([0] * (n_samples // 2) + [1] * (n_samples // 2))  # 0 = healthy, 1 = diseased
 
# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
 
# Build a simple CNN model
model = models.Sequential([
    layers.Input(shape=(image_size, image_size, 1)),
    layers.Conv2D(32, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='sigmoid')  # Binary output
])
 
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=5, batch_size=32, verbose=0)
 
# Evaluate
loss, acc = model.evaluate(X_test, y_test)
print(f"✅ Plant Disease Detection Accuracy: {acc:.4f}")
 
# Predict for 5 leaf images
preds = (model.predict(X_test[:5]) > 0.5).astype(int).flatten()
for i in range(5):
    status = "🌿 Healthy" if preds[i] == 0 else "⚠️ Diseased"
    actual = "🌿" if y_test[i] == 0 else "⚠️"
    print(f"Leaf {i+1}: Predicted = {status} | Actual = {actual}")