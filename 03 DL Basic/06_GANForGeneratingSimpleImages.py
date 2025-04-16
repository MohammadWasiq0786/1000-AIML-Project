"""
Project 86. GAN for generating simple images
Description:
A Generative Adversarial Network (GAN) consists of two neural networks — a Generator and a Discriminator — that compete in a zero-sum game. The generator learns to produce realistic data (e.g., images), while the discriminator learns to distinguish between real and fake data. In this project, we implement a basic GAN using Keras to generate handwritten digits similar to MNIST.

About:
🎨 What This Project Demonstrates:
Implements a basic GAN from scratch using Keras

Generates handwritten-style digits from noise vectors

Demonstrates the generator-discriminator adversarial loop
"""

# Install if not already: pip install tensorflow
 
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LeakyReLU, Reshape, Flatten
from tensorflow.keras.optimizers import Adam
 
# Load and preprocess MNIST dataset
(x_train, _), (_, _) = mnist.load_data()
x_train = x_train.astype('float32') / 127.5 - 1  # Normalize to [-1, 1]
x_train = x_train.reshape((-1, 28 * 28))
 
# Define dimensions
latent_dim = 100
 
# Build the Generator
generator = Sequential([
    Dense(128, input_dim=latent_dim),
    LeakyReLU(0.2),
    Dense(256),
    LeakyReLU(0.2),
    Dense(784, activation='tanh'),
    Reshape((28, 28))
])
 
# Build the Discriminator
discriminator = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(256),
    LeakyReLU(0.2),
    Dense(1, activation='sigmoid')
])
discriminator.compile(optimizer=Adam(0.0002, 0.5), loss='binary_crossentropy', metrics=['accuracy'])
 
# Combine into GAN
discriminator.trainable = False
gan = Sequential([generator, discriminator])
gan.compile(optimizer=Adam(0.0002, 0.5), loss='binary_crossentropy')
 
# Training loop
def train_gan(epochs=10000, batch_size=64, sample_interval=2000):
    for epoch in range(epochs):
        # Select a random batch of real images
        idx = np.random.randint(0, x_train.shape[0], batch_size)
        real_imgs = x_train[idx]
 
        # Generate fake images
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        fake_imgs = generator.predict(noise, verbose=0)
 
        # Labels for real and fake images
        real_y = np.ones((batch_size, 1))
        fake_y = np.zeros((batch_size, 1))
 
        # Train discriminator
        d_loss_real = discriminator.train_on_batch(real_imgs, real_y)
        d_loss_fake = discriminator.train_on_batch(fake_imgs, fake_y)
 
        # Train generator via GAN (discriminator frozen)
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        g_loss = gan.train_on_batch(noise, real_y)
 
        # Print progress
        if epoch % sample_interval == 0:
            print(f"Epoch {epoch} | D Loss: {np.mean([d_loss_real[0], d_loss_fake[0]]):.4f} | G Loss: {g_loss:.4f}")
            sample_images(generator, epoch)
 
# Function to generate and display images
def sample_images(generator, epoch, n=5):
    noise = np.random.normal(0, 1, (n * n, latent_dim))
    gen_imgs = generator.predict(noise, verbose=0)
    gen_imgs = 0.5 * gen_imgs + 0.5  # Rescale to [0, 1]
 
    plt.figure(figsize=(n, n))
    for i in range(n * n):
        plt.subplot(n, n, i + 1)
        plt.imshow(gen_imgs[i], cmap='gray')
        plt.axis('off')
    plt.suptitle(f"Generated Images at Epoch {epoch}")
    plt.tight_layout()
    plt.show()
 
# Train the GAN
train_gan(epochs=10000, batch_size=64, sample_interval=2000)