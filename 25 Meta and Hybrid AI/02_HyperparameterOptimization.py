"""
Project 962: Hyperparameter Optimization
Description
Hyperparameter optimization is the process of tuning the hyperparameters of a machine learning model (e.g., learning rate, batch size, number of layers) to improve performance. In this project, we will use random search or grid search to optimize hyperparameters for a neural network model.

Key Concepts Covered:
Hyperparameter Search Space: Defining a range of values for hyperparameters like learning rate, batch size, number of epochs, and units.

Random Search: A simple method to randomly sample combinations of hyperparameters, train the model, and track the best-performing configuration.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
 
# Load CIFAR-10 dataset for image classification
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0  # Normalize to [0, 1]
y_train, y_test = to_categorical(y_train, 10), to_categorical(y_test, 10)  # One-hot encoding
 
# Define the hyperparameter search space
hyperparameters = {
    'learning_rate': [0.001, 0.01, 0.1],  # Learning rates to try
    'batch_size': [32, 64, 128],           # Batch sizes
    'epochs': [10, 20, 30],                # Number of epochs
    'units': [64, 128, 256]                # Number of units in the dense layer
}
 
# Function to create a model based on the hyperparameters
def create_model(learning_rate, batch_size, units):
    model = models.Sequential([
        layers.Conv2D(units, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(units, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model
 
# Random Search for hyperparameters
best_accuracy = 0
best_params = None
 
# Randomly choose hyperparameters and train
for _ in range(5):  # Randomly sample 5 combinations
    learning_rate = np.random.choice(hyperparameters['learning_rate'])
    batch_size = np.random.choice(hyperparameters['batch_size'])
    epochs = np.random.choice(hyperparameters['epochs'])
    units = np.random.choice(hyperparameters['units'])
    
    print(f"Training model with learning_rate={learning_rate}, batch_size={batch_size}, epochs={epochs}, units={units}...")
    
    # Create and train the model
    model = create_model(learning_rate, batch_size, units)
    model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(x_test, y_test), verbose=0)
    
    # Evaluate the model
    _, accuracy = model.evaluate(x_test, y_test)
    
    # Track the best model
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_params = (learning_rate, batch_size, epochs, units)
    
    print(f"Model accuracy: {accuracy:.4f}")
 
print(f"Best Model Hyperparameters: {best_params}")
print(f"Best Accuracy: {best_accuracy:.4f}")