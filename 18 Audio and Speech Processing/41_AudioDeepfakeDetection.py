"""
Project 720: Audio Deepfake Detection
Description:
Audio deepfake detection focuses on identifying synthetic audio generated by deep learning models, which mimics real human speech. Deepfake audio can be used for malicious purposes, such as impersonating someone's voice, spreading misinformation, or creating fake news. In this project, we will implement a deepfake detection system that can distinguish between real and fake audio by analyzing speech characteristics using machine learning techniques.

For this project, we will extract spectrograms from both real and fake audio samples and use a Convolutional Neural Network (CNN) to classify the audio as either real or fake. We'll use Librosa for feature extraction and TensorFlow for building the neural network model.

Required Libraries:
pip install librosa tensorflow numpy matplotlib

Explanation:
Feature Extraction: We extract Mel-spectrogram features from both real and fake audio files. These features represent the time-frequency characteristics of the audio signal, which are important for detecting audio manipulations.

Deepfake Detection Model (CNN): We build a CNN-based model that takes the Mel-spectrogram as input and outputs a binary classification: 0 (real) or 1 (fake). The CNN is trained to distinguish between real and deepfake audio based on the spectrogram features.

Training: The model is trained using labeled audio samples (real and fake) to learn the differences between genuine and synthesized speech. It uses a binary cross-entropy loss function for training.

Testing: After training, we use the model to test a new audio file and predict if it is real or fake.

This project provides a basic audio deepfake detection system. For better results, you can experiment with larger datasets, data augmentation techniques, and more sophisticated models like GANs or LSTMs.
"""

import librosa
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
 
# 1. Load the audio file
def load_audio(file_path):
    audio, sr = librosa.load(file_path, sr=None)
    return audio, sr
 
# 2. Extract the spectrogram (Mel-spectrogram)
def extract_spectrogram(audio, sr, n_mfcc=13, n_mels=128, hop_length=512):
    # Convert to Mel Spectrogram
    mel_spectrogram = librosa.feature.melspectrogram(audio, sr=sr, n_mfcc=n_mfcc, n_mels=n_mels, hop_length=hop_length)
    spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
    return spectrogram_db
 
# 3. Build the deepfake detection model (CNN-based)
def build_deepfake_detection_model(input_shape):
    model = tf.keras.Sequential()
    model.add(layers.InputLayer(input_shape=input_shape))
    
    # Convolutional layers to extract features from spectrogram
    model.add(layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    
    # Flatten and add fully connected layers
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))  # Output: 0 (real) or 1 (fake)
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model
 
# 4. Train the model on real and fake audio samples
def train_model(model, real_audio_files, fake_audio_files, epochs=10, batch_size=32):
    X = []
    y = []
    
    # Extract features from real audio files
    for file in real_audio_files:
        audio, sr = load_audio(file)
        spectrogram = extract_spectrogram(audio, sr)
        X.append(spectrogram)
        y.append(0)  # Label 0 for real audio
    
    # Extract features from fake audio files
    for file in fake_audio_files:
        audio, sr = load_audio(file)
        spectrogram = extract_spectrogram(audio, sr)
        X.append(spectrogram)
        y.append(1)  # Label 1 for fake audio
    
    X = np.array(X)
    y = np.array(y)
    
    # Reshape X to fit the CNN input shape (add channel dimension)
    X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)
    
    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train the model
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test))
 
# 5. Example usage
real_audio_files = ['path_to_real_audio_1.wav', 'path_to_real_audio_2.wav']  # Replace with paths to real audio files
fake_audio_files = ['path_to_fake_audio_1.wav', 'path_to_fake_audio_2.wav']  # Replace with paths to fake audio files
 
# Build the deepfake detection model
model = build_deepfake_detection_model(input_shape=(128, 128, 1))  # Assuming 128x128 spectrogram input
 
# Train the model on the real and fake audio datasets
train_model(model, real_audio_files, fake_audio_files)
 
# Example test (you can test with new audio files)
test_audio_file = "path_to_test_audio.wav"  # Replace with the test audio file path
audio, sr = load_audio(test_audio_file)
spectrogram = extract_spectrogram(audio, sr)
spectrogram = spectrogram.reshape(1, spectrogram.shape[0], spectrogram.shape[1], 1)  # Reshape for CNN input
 
# Predict if the test audio is real or fake
prediction = model.predict(spectrogram)
if prediction > 0.5:
    print("The audio is fake.")
else:
    print("The audio is real.")