"""
Project 769: Federated Learning Simulation
Description
Federated learning is a privacy-preserving approach where multiple devices collaboratively train a model without sharing raw data. Each device trains on local data and shares only model updates (gradients or weights) with a central server. We'll simulate this in Python by splitting a dataset across clients, training locally, and averaging their weights—a basic version of Federated Averaging (FedAvg).

This simulates one round of FedAvg with 3 clients. In real federated systems like TensorFlow Federated, the process is automated with more security, encryption, and client-device orchestration.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
import copy
 
# Load and preprocess MNIST data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0
x_train = x_train[..., tf.newaxis]
x_test = x_test[..., tf.newaxis]
 
# Split the training data among 3 simulated clients
client_data = np.array_split(x_train, 3)
client_labels = np.array_split(y_train, 3)
 
# Define a small CNN model
def create_model():
    model = models.Sequential([
        layers.Conv2D(16, 3, activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(32, activation='relu'),
        layers.Dense(10)
    ])
    return model
 
# Compile model function
def compile_model(model):
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
 
# Create the global model
global_model = create_model()
compile_model(global_model)
 
# Simulate 1 round of federated averaging
local_weights = []
 
# Each client trains on their own local data
for i in range(3):
    print(f"Training client {i+1}...")
    client_model = create_model()
    compile_model(client_model)
    
    # Initialize client model with global weights
    client_model.set_weights(global_model.get_weights())
 
    # Train on local data
    client_model.fit(client_data[i], client_labels[i], epochs=1, batch_size=32, verbose=0)
 
    # Save local weights
    local_weights.append(client_model.get_weights())
 
# Average the local weights
new_weights = []
for weights in zip(*local_weights):
    new_weights.append(np.mean(weights, axis=0))
 
# Update global model
global_model.set_weights(new_weights)
 
# Evaluate the updated global model
loss, acc = global_model.evaluate(x_test, y_test)
print(f"\n✅ Federated Learning Simulation Complete — Accuracy: {acc:.4f}")