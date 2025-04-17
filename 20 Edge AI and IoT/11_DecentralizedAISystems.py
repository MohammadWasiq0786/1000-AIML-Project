"""
Project 771: Decentralized AI Systems
Description
Decentralized AI systems operate without a central server, instead relying on peer-to-peer communication and coordination among edge devices or nodes. This ensures privacy, fault-tolerance, and scalability. A simplified simulation involves multiple nodes training independently and exchanging models with each other instead of a central aggregator (as in federated learning). We simulate decentralized model averaging among peers.

This simulates peer-based learning and synchronization without a server. In real-world scenarios, decentralized systems may use blockchain, gossip protocols, or Swarm learning for coordination, often applied in sensitive domains like healthcare and finance.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
import copy
 
# Load and preprocess MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
x_train = x_train[..., tf.newaxis]
x_test = x_test[..., tf.newaxis]
 
# Split training data among 3 decentralized nodes (peers)
peer_data = np.array_split(x_train, 3)
peer_labels = np.array_split(y_train, 3)
 
# Define a small CNN model
def build_model():
    model = models.Sequential([
        layers.Conv2D(16, 3, activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(32, activation='relu'),
        layers.Dense(10)
    ])
    return model
 
# Compile helper
def compile_model(model):
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
 
# Each node trains its own local model
local_models = []
for i in range(3):
    print(f"Peer {i+1}: training on local data...")
    model = build_model()
    compile_model(model)
    model.fit(peer_data[i], peer_labels[i], epochs=1, batch_size=32, verbose=0)
    local_models.append(model)
 
# Simulate peer-to-peer model exchange and average weights
averaged_weights = []
for weights in zip(*[m.get_weights() for m in local_models]):
    averaged_weights.append(np.mean(weights, axis=0))
 
# Each node updates its model with the averaged weights (no central aggregator)
for i in range(3):
    local_models[i].set_weights(averaged_weights)
 
# Evaluate one of the peer models
loss, acc = local_models[0].evaluate(x_test, y_test)
print(f"\n✅ Decentralized AI Simulation Accuracy (Peer 1): {acc:.4f}")