"""
Project 111. Image denoising autoencoder
Description:
An Image Denoising Autoencoder is a neural network that learns to remove noise from images. It’s trained to reconstruct clean images from noisy inputs. In this project, we build a simple convolutional autoencoder using TensorFlow/Keras and train it on the MNIST dataset corrupted with synthetic noise.

About:
🧠 What This Project Demonstrates:
Trains an autoencoder neural network for noise reduction

Uses convolutional layers to learn spatial patterns

Applies to real-world tasks like photo cleanup and restoration
"""

# Install if not already: pip install tensorflow matplotlib numpy
 
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
 
# Load MNIST data and normalize
(x_train, _), (x_test, _) = mnist.load_data()
x_train = x_train.astype("float32") / 255.
x_test = x_test.astype("float32") / 255.
 
# Reshape to (num_samples, 28, 28, 1)
x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))
x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))
 
# Add random noise to images
noise_factor = 0.5
x_train_noisy = x_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape)
x_test_noisy = x_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape)
 
# Clip pixel values to [0, 1]
x_train_noisy = np.clip(x_train_noisy, 0., 1.)
x_test_noisy = np.clip(x_test_noisy, 0., 1.)
 
# Build the autoencoder
input_img = Input(shape=(28, 28, 1))
 
# Encoder
x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
encoded = MaxPooling2D((2, 2), padding='same')(x)
 
# Decoder
x = Conv2D(16, (3, 3), activation='relu', padding='same')(encoded)
x = UpSampling2D((2, 2))(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
 
# Model
autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
autoencoder.fit(x_train_noisy, x_train, epochs=10, batch_size=128, shuffle=True, validation_split=0.1, verbose=1)
 
# Predict on test set
decoded_imgs = autoencoder.predict(x_test_noisy)
 
# Display original noisy and denoised
n = 10
plt.figure(figsize=(20, 4))
 
for i in range(n):
    # Noisy input
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test_noisy[i].reshape(28, 28), cmap="gray")
    plt.title("Noisy")
    plt.axis("off")
 
    # Denoised output
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(28, 28), cmap="gray")
    plt.title("Denoised")
    plt.axis("off")
 
plt.suptitle("Image Denoising Autoencoder - MNIST")
plt.tight_layout()
plt.show()