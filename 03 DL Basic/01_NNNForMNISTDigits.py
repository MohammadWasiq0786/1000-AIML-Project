"""
Project 81. Neural network for MNIST digit recognition
Description:
The MNIST digit recognition task involves classifying handwritten digits (0–9) from grayscale images. In this project, we build a fully connected neural network (Multilayer Perceptron) using TensorFlow/Keras to recognize digits from the MNIST dataset.

About:
🔢 What This Project Demonstrates:
Builds a basic feedforward neural network for image classification

Uses MNIST, a benchmark dataset in computer vision

Trains with categorical cross-entropy and evaluates accuracy
"""

# Install if not already: pip install tensorflow
 
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical
 
# Load MNIST data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
 
# Normalize input data
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
 
# One-hot encode labels
y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)
 
# Build the neural network model
model = Sequential([
    Flatten(input_shape=(28, 28)),           # Flatten 28x28 images to 784
    Dense(128, activation='relu'),           # First hidden layer
    Dense(64, activation='relu'),            # Second hidden layer
    Dense(10, activation='softmax')          # Output layer for 10 classes
])
 
# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
 
# Train the model
model.fit(x_train, y_train, epochs=10, batch_size=128, validation_split=0.2, verbose=1)
 
# Evaluate the model
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"\nTest Accuracy: {test_acc:.4f}")
 
# Predict on a few test samples
predictions = model.predict(x_test[:10])
 
# Plot some test results
plt.figure(figsize=(12, 2))
for i in range(10):
    plt.subplot(1, 10, i+1)
    plt.imshow(x_test[i], cmap='gray')
    plt.title(np.argmax(predictions[i]))
    plt.axis('off')
plt.suptitle("MNIST Predictions (First 10 Test Samples)")
plt.tight_layout()
plt.show()