"""
Project 201. Image classification for common objects
Description:
Image Classification is the task of assigning a label to an image from a predefined set of classes. In this project, we build a simple image classifier for common everyday objects using a Convolutional Neural Network (CNN) with TensorFlow/Keras, trained on a dataset like CIFAR-10 or your own custom data.

About:
🧠 What This Project Demonstrates:
Trains a CNN model from scratch to classify images

Uses the CIFAR-10 dataset with 10 common object categories
"""

# Install if not already: pip install tensorflow
 
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
 
# Load CIFAR-10 dataset (60,000 32x32 color images in 10 classes)
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
 
# Normalize image data to 0–1
x_train, x_test = x_train / 255.0, x_test / 255.0
 
# One-hot encode labels
y_train_cat = to_categorical(y_train, 10)
y_test_cat = to_categorical(y_test, 10)
 
# Class names for CIFAR-10
class_names = ["airplane", "automobile", "bird", "cat", "deer", 
               "dog", "frog", "horse", "ship", "truck"]
 
# Build CNN model
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(32,32,3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')  # 10 classes
])
 
# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
 
# Train the model
print("🧠 Training the model...")
model.fit(x_train, y_train_cat, epochs=10, validation_data=(x_test, y_test_cat), batch_size=64)
 
# Evaluate on test set
loss, acc = model.evaluate(x_test, y_test_cat)
print(f"\n✅ Test Accuracy: {acc*100:.2f}%")
 
# Predict sample image
import numpy as np
sample_idx = 12
sample_image = x_test[sample_idx]
prediction = model.predict(np.expand_dims(sample_image, axis=0))
predicted_label = class_names[np.argmax(prediction)]
 
plt.imshow(sample_image)
plt.title(f"Predicted: {predicted_label}")
plt.axis('off')
plt.show()