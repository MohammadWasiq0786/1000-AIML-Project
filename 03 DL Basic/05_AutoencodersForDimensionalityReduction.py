"""
Project 85. Autoencoders for dimensionality reduction
Description:
Autoencoders are neural networks that learn to compress input data into a lower-dimensional representation and then reconstruct it. This ability makes them powerful tools for dimensionality reduction, often outperforming traditional techniques like PCA in nonlinear data. In this project, we build a simple autoencoder using Keras to reduce dimensions of MNIST digit data.

About:
📉 What This Project Demonstrates:
Compresses high-dimensional image data into a lower-dimensional latent space

Learns nonlinear representations using deep neural networks

Visualizes compressed embeddings for exploratory analysis
"""

# Install if not already: pip install tensorflow
 
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.utils import plot_model
 
# Load MNIST dataset
(x_train, _), (x_test, _) = mnist.load_data()
x_train = x_train.astype("float32") / 255.
x_test = x_test.astype("float32") / 255.
x_train = x_train.reshape((len(x_train), -1))  # Flatten to 784
x_test = x_test.reshape((len(x_test), -1))
 
# Define dimensions
input_dim = x_train.shape[1]  # 784
encoding_dim = 32             # Desired lower dimension
 
# Build autoencoder
input_img = Input(shape=(input_dim,))
encoded = Dense(128, activation='relu')(input_img)
encoded = Dense(64, activation='relu')(encoded)
encoded_output = Dense(encoding_dim, activation='relu')(encoded)
 
decoded = Dense(64, activation='relu')(encoded_output)
decoded = Dense(128, activation='relu')(decoded)
decoded_output = Dense(input_dim, activation='sigmoid')(decoded)
 
# Full autoencoder model
autoencoder = Model(input_img, decoded_output)
 
# Encoder model for reduced feature extraction
encoder = Model(input_img, encoded_output)
 
# Compile and train
autoencoder.compile(optimizer='adam', loss='mse')
autoencoder.fit(x_train, x_train, epochs=20, batch_size=256, shuffle=True,
                validation_data=(x_test, x_test), verbose=1)
 
# Encode the test data into reduced features
encoded_imgs = encoder.predict(x_test)
 
# Visualize encoded data using first 2 dimensions
plt.figure(figsize=(8, 6))
plt.scatter(encoded_imgs[:, 0], encoded_imgs[:, 1], alpha=0.5, s=2)
plt.title("2D Projection of MNIST using Autoencoder")
plt.xlabel("Encoded Feature 1")
plt.ylabel("Encoded Feature 2")
plt.grid(True)
plt.tight_layout()
plt.show()