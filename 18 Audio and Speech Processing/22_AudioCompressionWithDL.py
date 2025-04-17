"""
Project 701: Audio Compression with Deep Learning
Description:
Audio compression is the process of reducing the size of an audio file without sacrificing too much quality. It is used in applications like streaming and storage to minimize file size while maintaining a good listening experience. In this project, we will implement an audio compression system using deep learning techniques, such as autoencoders, to learn efficient representations of audio and compress it.

Here, we will use a simple autoencoder for audio compression. The autoencoder will learn a compressed representation of the audio, and then reconstruct the audio from this compressed representation. The model will be trained using the mean squared error loss function, which measures the difference between the original and reconstructed audio.

Explanation:
In this audio compression project, we use a simple autoencoder for learning a compressed representation of the audio. The autoencoder consists of an encoder that reduces the dimensionality of the input and a decoder that reconstructs the original audio from the compressed representation. The autoencoder is trained using mean squared error loss to minimize the reconstruction error between the original and reconstructed audio.

This basic implementation can be improved by using more sophisticated architectures such as variational autoencoders (VAE) or convolutional autoencoders for better audio compression.
"""

import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt
 
# 1. Load the audio file
def load_audio(file_path):
    audio, sr = librosa.load(file_path, sr=None)
    return audio, sr
 
# 2. Preprocess the audio (normalize and reshape)
def preprocess_audio(audio, target_length=16000):
    # Normalize the audio signal to a range of [-1, 1]
    audio = audio / np.max(np.abs(audio))
    
    # Trim or pad the audio to match the target length
    if len(audio) > target_length:
        audio = audio[:target_length]
    else:
        audio = np.pad(audio, (0, target_length - len(audio)))
 
    return audio
 
# 3. Define the autoencoder model for audio compression
def build_autoencoder(input_shape):
    model = tf.keras.Sequential()
 
    # Encoder: Reduce dimensionality
    model.add(layers.InputLayer(input_shape=input_shape))
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(128, activation='relu'))  # Compressed representation
 
    # Decoder: Reconstruct the original audio
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(input_shape[0], activation='sigmoid'))  # Output layer with same size as input
 
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model
 
# 4. Train the autoencoder model
def train_autoencoder(model, audio_data, epochs=10):
    # Reshape data for training
    audio_data = np.reshape(audio_data, (-1, len(audio_data)))  # Flatten the audio into vectors
 
    # Train the model on the audio data
    model.fit(audio_data, audio_data, epochs=epochs, batch_size=16)
 
# 5. Compress and reconstruct the audio
def compress_and_reconstruct(model, audio):
    compressed_audio = model.predict(np.reshape(audio, (-1, len(audio))))  # Compress and reconstruct
    return compressed_audio[0]
 
# 6. Example usage
audio_file = "path_to_audio.wav"  # Replace with the path to your audio file
 
# Load and preprocess the audio
audio, sr = load_audio(audio_file)
processed_audio = preprocess_audio(audio)
 
# Build and train the autoencoder model
model = build_autoencoder(input_shape=(len(processed_audio),))
train_autoencoder(model, processed_audio, epochs=50)
 
# Compress and reconstruct the audio
reconstructed_audio = compress_and_reconstruct(model, processed_audio)
 
# Plot the original and reconstructed audio for comparison
plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
plt.plot(processed_audio)
plt.title("Original Audio")
 
plt.subplot(2, 1, 2)
plt.plot(reconstructed_audio)
plt.title("Reconstructed Audio (After Compression)")
 
plt.tight_layout()
plt.show()