"""
Project 95. Handwriting recognition system
Description:
A handwriting recognition system interprets handwritten characters or digits from scanned images. In this project, we build a convolutional neural network (CNN) using Keras to recognize handwritten digits from the MNIST dataset, which serves as a great foundation for building full OCR systems.

About:
✍️ What This Project Demonstrates:
Uses a CNN to recognize handwritten digits

Trains on MNIST, a classic dataset for OCR tasks

Outputs digit predictions with high accuracy
"""

# Install if not already: pip install tensorflow matplotlib
 
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
 
# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
 
# Preprocess data: reshape and normalize
x_train = x_train.reshape(-1, 28, 28, 1).astype("float32") / 255.0
x_test = x_test.reshape(-1, 28, 28, 1).astype("float32") / 255.0
 
# One-hot encode target labels
y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)
 
# Build CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2)),
 
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
 
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')  # Output layer for digits 0–9
])
 
# Compile model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
 
# Train model
model.fit(x_train, y_train, epochs=5, batch_size=64, validation_split=0.2, verbose=1)
 
# Evaluate model
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
print(f"\nTest Accuracy: {test_accuracy:.4f}")
 
# Predict and display some results
predictions = model.predict(x_test[:10])
 
plt.figure(figsize=(12, 2))
for i in range(10):
    plt.subplot(1, 10, i + 1)
    plt.imshow(x_test[i].reshape(28, 28), cmap='gray')
    plt.title(np.argmax(predictions[i]))
    plt.axis('off')
plt.suptitle("Handwriting Recognition - Predictions on MNIST")
plt.tight_layout()
plt.show()