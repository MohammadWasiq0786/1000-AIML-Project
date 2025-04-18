"""
Project 863: Recycling Sorting System
Description
A recycling sorting system uses computer vision to classify waste materials (plastic, metal, paper, organic, etc.) for automated or assisted recycling. In this project, we simulate image inputs and build a CNN-based multi-class classifier to predict the waste category of an item from its visual features.

✅ This model can be deployed in:

Smart sorting bins

Recycling plant conveyor systems

Educational tools to teach waste segregation
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
 
# Simulate image data (replace with real dataset like TrashNet for production)
np.random.seed(42)
n_samples = 1000
image_size = 64
n_classes = 4  # 0: Plastic, 1: Metal, 2: Paper, 3: Organic
 
# Create dummy image data for each class
def generate_waste_images(label, pattern_shift):
    images = np.random.normal(loc=0.3 + pattern_shift, scale=0.1, size=(n_samples // n_classes, image_size, image_size, 1))
    labels = [label] * (n_samples // n_classes)
    return images, labels
 
plastic_imgs, plastic_labels = generate_waste_images(0, 0.05)
metal_imgs, metal_labels = generate_waste_images(1, 0.15)
paper_imgs, paper_labels = generate_waste_images(2, 0.25)
organic_imgs, organic_labels = generate_waste_images(3, -0.05)
 
# Combine dataset
X = np.vstack([plastic_imgs, metal_imgs, paper_imgs, organic_imgs])
y = np.array(plastic_labels + metal_labels + paper_labels + organic_labels)
 
# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
 
# Build CNN model for classification
model = models.Sequential([
    layers.Input(shape=(image_size, image_size, 1)),
    layers.Conv2D(32, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(n_classes, activation='softmax')  # Output: 4 categories
])
 
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=5, batch_size=32, verbose=0)
 
# Evaluate model
loss, acc = model.evaluate(X_test, y_test)
print(f"✅ Recycling Sorting Accuracy: {acc:.4f}")
 
# Predict and show class for a few items
preds = np.argmax(model.predict(X_test[:5]), axis=1)
class_map = {0: "♳ Plastic", 1: "♴ Metal", 2: "📄 Paper", 3: "🍎 Organic"}
 
for i in range(5):
    print(f"Item {i+1}: Predicted = {class_map[preds[i]]}, Actual = {class_map[y_test[i]]}")