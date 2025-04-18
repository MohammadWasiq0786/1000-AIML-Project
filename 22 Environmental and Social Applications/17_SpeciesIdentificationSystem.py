"""
Project 857: Species Identification System
Description
A species identification system classifies animals or plants based on images, audio, or environmental features. In this project, we simulate a vision-based classifier that takes in an image of an animal and predicts its species (e.g., tiger, deer, elephant, etc.), using a CNN model.

✅ This model supports:

Automated camera trap analysis

Mobile wildlife identification tools

Integration with conservation databases (e.g., iNaturalist, GBIF)
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
 
# Simulate small grayscale image data (e.g., from camera traps)
np.random.seed(42)
n_samples = 1200
image_size = 64
n_classes = 3  # e.g., 0 = Tiger, 1 = Deer, 2 = Elephant
 
# Simulate species images with slight pattern differences (you'd replace this with real images)
def generate_species_images(label, pattern_shift):
    base = np.random.normal(loc=0.4 + pattern_shift, scale=0.1, size=(n_samples // n_classes, image_size, image_size, 1))
    return base, [label] * (n_samples // n_classes)
 
tiger_imgs, tiger_labels = generate_species_images(0, 0.1)
deer_imgs, deer_labels = generate_species_images(1, 0.0)
elephant_imgs, elephant_labels = generate_species_images(2, -0.1)
 
# Combine dataset
X = np.vstack([tiger_imgs, deer_imgs, elephant_imgs])
y = np.array(tiger_labels + deer_labels + elephant_labels)
 
# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
 
# Build CNN model for image classification
model = models.Sequential([
    layers.Input(shape=(image_size, image_size, 1)),
    layers.Conv2D(32, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(n_classes, activation='softmax')  # Multi-class output
])
 
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=5, batch_size=32, verbose=0)
 
# Evaluate model
loss, acc = model.evaluate(X_test, y_test)
print(f"✅ Species Identification Accuracy: {acc:.4f}")
 
# Predict for 5 animal images
preds = np.argmax(model.predict(X_test[:5]), axis=1)
species_map = {0: "🐯 Tiger", 1: "🦌 Deer", 2: "🐘 Elephant"}
 
for i in range(5):
    print(f"Image {i+1}: Predicted = {species_map[preds[i]]}, Actual = {species_map[y_test[i]]}")