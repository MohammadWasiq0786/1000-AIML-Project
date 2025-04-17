"""
Project 770: Split Learning Implementation
Description
Split learning is a collaborative training method where a model is split between a client (e.g., edge device) and a server (e.g., cloud). The client computes forward passes up to a certain layer (called the cut layer) and sends the activations to the server, which continues the forward and backward pass. This approach improves data privacy while offloading computation.

We'll simulate this with a split CNN where the client runs the first few layers, and the server runs the rest.

This code mimics the flow of split learning where the client and server only share intermediate activations, not raw data—enhancing privacy and efficiency in real-world edge-cloud collaboration.
"""

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
import numpy as np
 
# Load and preprocess MNIST
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
x_train = x_train[..., tf.newaxis]
x_test = x_test[..., tf.newaxis]
 
# Define client-side model (up to the cut layer)
def create_client_model():
    inputs = tf.keras.Input(shape=(28, 28, 1))
    x = layers.Conv2D(16, 3, activation='relu')(inputs)
    x = layers.MaxPooling2D()(x)
    return tf.keras.Model(inputs=inputs, outputs=x, name="client_model")
 
# Define server-side model (from cut layer onward)
def create_server_model():
    inputs = tf.keras.Input(shape=(13, 13, 16))  # Shape after client's output
    x = layers.Flatten()(inputs)
    x = layers.Dense(32, activation='relu')(x)
    x = layers.Dense(10)(x)
    return tf.keras.Model(inputs=inputs, outputs=x, name="server_model")
 
# Instantiate models
client_model = create_client_model()
server_model = create_server_model()
 
# Loss function and optimizer
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.Adam()
 
# Training loop for split learning (single epoch for brevity)
batch_size = 64
train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size)
 
for step, (images, labels) in enumerate(train_ds):
    with tf.GradientTape(persistent=True) as tape:
        # CLIENT: Forward pass up to cut layer
        client_output = client_model(images, training=True)
 
        # SERVER: Continue forward pass
        logits = server_model(client_output, training=True)
 
        # Compute loss
        loss = loss_fn(labels, logits)
 
    # Backpropagation
    server_grads = tape.gradient(loss, server_model.trainable_variables)
    client_grads = tape.gradient(loss, client_model.trainable_variables)
 
    # Apply gradients
    optimizer.apply_gradients(zip(server_grads, server_model.trainable_variables))
    optimizer.apply_gradients(zip(client_grads, client_model.trainable_variables))
 
    if step % 100 == 0:
        print(f"Step {step} — Loss: {loss:.4f}")
 
# Evaluate on test set
client_out = client_model(x_test, training=False)
preds = server_model(client_out, training=False)
accuracy = tf.keras.metrics.sparse_categorical_accuracy(y_test, preds)
print(f"\n✅ Split Learning Test Accuracy: {tf.reduce_mean(accuracy):.4f}")