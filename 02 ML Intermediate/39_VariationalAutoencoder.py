"""
Project 79. Variational autoencoder
Description:
A Variational Autoencoder (VAE) is a type of generative neural network that learns to encode input data into a latent space and then decode it back to reconstruct the original input. VAEs are powerful for dimensionality reduction, image generation, and anomaly detection. In this project, we build a simple VAE using Keras and apply it to the MNIST dataset.

About:
🌀 What This Project Demonstrates:
Builds a Variational Autoencoder for MNIST data

Learns probabilistic latent variables for generative modeling

Visualizes the latent space to show clustering of digits
"""

# Install if not already: pip install tensorflow
 
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from tensorflow.keras.losses import binary_crossentropy
 
# Load MNIST dataset
(x_train, _), (x_test, _) = mnist.load_data()
x_train = x_train.astype("float32") / 255.
x_test = x_test.astype("float32") / 255.
x_train = np.reshape(x_train, (-1, 28 * 28))
x_test = np.reshape(x_test, (-1, 28 * 28))
 
# VAE parameters
input_dim = 784
latent_dim = 2
 
# Encoder
inputs = layers.Input(shape=(input_dim,))
h = layers.Dense(256, activation='relu')(inputs)
z_mean = layers.Dense(latent_dim)(h)
z_log_var = layers.Dense(latent_dim)(h)
 
# Sampling function
def sampling(args):
    z_mean, z_log_var = args
    epsilon = tf.random.normal(shape=(tf.shape(z_mean)[0], latent_dim))
    return z_mean + tf.exp(0.5 * z_log_var) * epsilon
 
z = layers.Lambda(sampling)([z_mean, z_log_var])
 
# Decoder
decoder_h = layers.Dense(256, activation='relu')
decoder_out = layers.Dense(input_dim, activation='sigmoid')
h_decoded = decoder_h(z)
outputs = decoder_out(h_decoded)
 
# VAE model
vae = models.Model(inputs, outputs)
 
# VAE loss
reconstruction_loss = binary_crossentropy(inputs, outputs) * input_dim
kl_loss = -0.5 * tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1)
vae_loss = tf.reduce_mean(reconstruction_loss + kl_loss)
 
vae.add_loss(vae_loss)
vae.compile(optimizer='adam')
vae.summary()
 
# Train the model
vae.fit(x_train, x_train, epochs=30, batch_size=128, validation_data=(x_test, x_test))
 
# Visualize latent space
encoder = models.Model(inputs, z_mean)
z_test = encoder.predict(x_test)
 
plt.figure(figsize=(8, 6))
plt.scatter(z_test[:, 0], z_test[:, 1], alpha=0.5)
plt.title("Latent Space Representation (VAE - MNIST)")
plt.xlabel("z[0]")
plt.ylabel("z[1]")
plt.grid(True)
plt.tight_layout()
plt.show()