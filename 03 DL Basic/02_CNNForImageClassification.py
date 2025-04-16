"""
Project 82. CNN for image classification
Description:
A Convolutional Neural Network (CNN) is a powerful deep learning architecture used for image classification and recognition. In this project, we build a CNN model using Keras to classify images from the CIFAR-10 dataset, which contains 60,000 32x32 color images across 10 classes (e.g., airplanes, cars, birds).

🧠 What This Project Demonstrates:
Builds a CNN architecture with Conv2D, MaxPooling, and Dropout

Uses CIFAR-10, a standard dataset for object classification

Evaluates model with accuracy and visualizes predictions
"""


# Install if not already: pip install tensorflow
 
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import numpy as np
 
# Load CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
 
# Normalize pixel values
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
 
# One-hot encode labels
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)
 
# Define CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    MaxPooling2D(pool_size=(2, 2)),
 
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
 
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')  # 10 classes
])
 
# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
 
# Train the model
model.fit(x_train, y_train, epochs=10, batch_size=64, validation_split=0.2, verbose=1)
 
# Evaluate the model
loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
print(f"\nTest Accuracy: {accuracy:.4f}")
 
# Show predictions on a few test images
class_names = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
predictions = model.predict(x_test[:10])
 
plt.figure(figsize=(12, 2))
for i in range(10):
    plt.subplot(1, 10, i + 1)
    plt.imshow(x_test[i])
    plt.title(class_names[np.argmax(predictions[i])])
    plt.axis('off')
plt.suptitle("CIFAR-10 Predictions (First 10 Samples)")
plt.tight_layout()
plt.show()