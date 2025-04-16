"""
Project 39. Simple autoencoder
Description:
An autoencoder is a type of artificial neural network used for learning efficient codings of input data in an unsupervised manner. It consists of two parts: an encoder that compresses the input and a decoder that reconstructs it. In this project, we build a simple autoencoder using Keras to compress and reconstruct the Iris dataset.

About:
🔍 What This Project Shows:
Compresses 4D data to 2D using an autoencoder

Learns a non-linear transformation

Visualizes learned embeddings and structure between classes
"""


# Install TensorFlow if not installed
# pip install tensorflow
 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
 
# Load and scale the Iris dataset
iris = load_iris()
X = iris.data
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
 
# ---- Define Autoencoder Architecture ----
input_dim = X_scaled.shape[1]
encoding_dim = 2  # Reduce to 2 dimensions
 
# Encoder
input_layer = Input(shape=(input_dim,))
encoded = Dense(encoding_dim, activation='relu')(input_layer)
 
# Decoder
decoded = Dense(input_dim, activation='sigmoid')(encoded)
 
# Autoencoder Model
autoencoder = Model(inputs=input_layer, outputs=decoded)
autoencoder.compile(optimizer=Adam(learning_rate=0.01), loss='mse')
 
# Train the autoencoder
history = autoencoder.fit(X_scaled, X_scaled, epochs=100, batch_size=16, verbose=0)
 
# Encoder model for extracting compressed representations
encoder = Model(inputs=input_layer, outputs=encoded)
encoded_data = encoder.predict(X_scaled)
 
# ---- Plot Encoded Data ----
plt.figure(figsize=(7, 5))
for i in range(3):
    plt.scatter(encoded_data[iris.target == i, 0], encoded_data[iris.target == i, 1], label=iris.target_names[i])
plt.title("2D Encoded Representation (Autoencoder)")
plt.xlabel("Encoded Dimension 1")
plt.ylabel("Encoded Dimension 2")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()