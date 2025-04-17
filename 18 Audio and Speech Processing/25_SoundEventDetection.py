"""
Project 704: Sound Event Detection
Description:
Sound event detection involves identifying and classifying specific sounds within an audio stream, such as detecting the sound of a dog barking, a doorbell ringing, or a car honking. This is important for applications like smart home systems, surveillance, and environmental monitoring. In this project, we will implement a sound event detection system that uses spectrograms and deep learning models (such as CNNs or LSTMs) to detect and classify different sound events.

We'll use spectrograms as input features for the CNN model. A spectrogram represents the time-frequency distribution of an audio signal and is commonly used for audio classification tasks. The system will classify sound events based on the audio signal's spectrogram.

Explanation:
In this sound event detection system, we extract spectrograms from the audio signal, which represent the frequency content of the audio over time. A Convolutional Neural Network (CNN) is then used to classify the spectrograms into different sound events (e.g., dog barking, car honking). The model is trained on a labeled dataset of audio clips, and it can be used to predict the presence of specific sound events in new audio files.

For real-world applications, you would need a dataset like UrbanSound8K or AudioSet, which contains labeled environmental sounds for training the model.
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
 
# 2. Extract spectrogram from the audio
def extract_spectrogram(audio, sr, n_fft=2048, hop_length=512, n_mels=128):
    # Convert audio to spectrogram using Mel-frequency scaling
    mel_spectrogram = librosa.feature.melspectrogram(audio, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)  # Convert to dB scale for better visualization
    return spectrogram_db
 
# 3. Build a simple CNN for sound event detection
def build_cnn_model(input_shape):
    model = tf.keras.Sequential()
    
    # CNN layers for feature extraction
    model.add(layers.InputLayer(input_shape=input_shape))
    model.add(layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    
    # Flatten and classify the sound event
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))  # 10 classes for sound events (can be adjusted)
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model
 
# 4. Train the model on a labeled sound event dataset (use a proper dataset for training)
def train_model(model, X_train, y_train, epochs=10):
    model.fit(X_train, y_train, epochs=epochs, batch_size=32)
 
# 5. Predict the sound event for a new audio file
def predict_sound_event(model, audio, sr):
    spectrogram = extract_spectrogram(audio, sr)
    spectrogram = np.expand_dims(spectrogram, axis=-1)  # Add channel dimension (needed for CNN)
    spectrogram = np.expand_dims(spectrogram, axis=0)  # Add batch dimension
    
    prediction = model.predict(spectrogram)
    predicted_class = np.argmax(prediction)
    return predicted_class
 
# 6. Visualize the spectrogram of the audio
def plot_spectrogram(spectrogram):
    plt.figure(figsize=(10, 6))
    plt.imshow(spectrogram, cmap='viridis', origin='lower', aspect='auto')
    plt.title("Spectrogram")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.colorbar(format="%+2.0f dB")
    plt.show()
 
# 7. Example usage
audio_file = "path_to_audio_file.wav"  # Replace with your audio file path
 
# Load the audio signal
audio, sr = load_audio(audio_file)
 
# Extract the spectrogram from the audio signal
spectrogram = extract_spectrogram(audio, sr)
 
# Plot the spectrogram
plot_spectrogram(spectrogram)
 
# Build and train the model (requires a dataset for real training)
model = build_cnn_model(input_shape=spectrogram.shape + (1,))  # Add channel dimension for CNN
# Train the model here with a real dataset (X_train, y_train)
 
# Predict the sound event for a new audio file
predicted_event = predict_sound_event(model, audio, sr)
print(f"Predicted sound event: {predicted_event}")