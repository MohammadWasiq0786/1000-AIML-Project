"""
Project 961: Neural Architecture Search (NAS)
Description
Neural Architecture Search (NAS) automates the process of designing the architecture of neural networks. It involves defining a search space for the network architecture, selecting an optimization strategy, and using search algorithms (e.g., reinforcement learning, evolutionary algorithms) to discover the best-performing architecture for a task.

Key Concepts Covered:
Search Space Definition: We define potential architectures such as the number of convolutional layers, number of units in the dense layer, and dropout rates.

Random Search: The simplest form of NAS, where we randomly select values from the search space and evaluate the resulting model.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
 
# Load CIFAR-10 dataset for image classification
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0  # Normalize to [0, 1]
y_train, y_test = to_categorical(y_train, 10), to_categorical(y_test, 10)  # One-hot encoding
 
# Define a basic model search space (in this case, layers and units)
search_space = {
    'conv_layers': [2, 3, 4],   # Number of convolution layers
    'units': [64, 128, 256],     # Number of units in each dense layer
    'dropout_rate': [0.2, 0.3, 0.4],  # Dropout rate
}
 
# Randomly choose architecture from search space
def create_model():
    conv_layers = np.random.choice(search_space['conv_layers'])
    units = np.random.choice(search_space['units'])
    dropout_rate = np.random.choice(search_space['dropout_rate'])
    
    model = models.Sequential()
    
    # Add convolutional layers
    for _ in range(conv_layers):
        model.add(layers.Conv2D(units, (3, 3), activation='relu', padding='same'))
        model.add(layers.MaxPooling2D((2, 2)))
    
    # Flatten and add dense layer with dropout
    model.add(layers.Flatten())
    model.add(layers.Dense(units, activation='relu'))
    model.add(layers.Dropout(dropout_rate))
    
    # Output layer
    model.add(layers.Dense(10, activation='softmax'))
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model
 
# Train the model using random search for 5 iterations
best_model = None
best_accuracy = 0
 
for i in range(5):
    print(f"Training model {i+1}...")
    model = create_model()
    
    # Train the model
    model.fit(x_train, y_train, epochs=5, batch_size=64, validation_data=(x_test, y_test), verbose=1)
    
    # Evaluate the model
    _, accuracy = model.evaluate(x_test, y_test)
    
    # Track the best model
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = model
    print(f"Model {i+1} Accuracy: {accuracy:.4f}")
 
print(f"Best Model Accuracy: {best_accuracy:.4f}")