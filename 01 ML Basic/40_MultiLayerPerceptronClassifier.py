"""
Project 40. Multi-layer perceptron classifier
Description:
A Multi-Layer Perceptron (MLP) is a class of feedforward artificial neural networks with one or more hidden layers. It can model complex relationships and is widely used for classification tasks. In this project, we implement an MLP classifier using Keras to classify the Iris dataset.

About:
🧠 What This Project Demonstrates:
Builds a fully connected neural network for multi-class classification

Trains and validates an MLP using backpropagation

Visualizes the accuracy curve to monitor performance
"""

# Install TensorFlow if needed
# pip install tensorflow
 
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelBinarizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import plot_model
import matplotlib.pyplot as plt
 
# Load and preprocess the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target
 
# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
 
# One-hot encode target labels
lb = LabelBinarizer()
y_encoded = lb.fit_transform(y)
 
# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.3, random_state=42)
 
# ---- Build MLP model ----
model = Sequential([
    Dense(10, input_shape=(4,), activation='relu'),  # First hidden layer
    Dense(8, activation='relu'),                     # Second hidden layer
    Dense(3, activation='softmax')                   # Output layer for 3 classes
])
 
# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
 
# Train the model
history = model.fit(X_train, y_train, epochs=100, batch_size=8, verbose=0, validation_split=0.2)
 
# Evaluate on test set
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy: {test_accuracy:.2f}")
 
# ---- Plot training history ----
plt.figure(figsize=(8, 4))
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title('MLP Classification Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()