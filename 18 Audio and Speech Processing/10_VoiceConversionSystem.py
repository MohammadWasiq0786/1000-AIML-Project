"""
Project 690: Voice Conversion System Using Simple Feature Mapping
Description:
Voice conversion (VC) involves transforming the speech signal of one person to sound like it is spoken by another person. This is commonly used in voice modification applications such as personalized assistants and speech synthesis. In this project, we will implement a simple voice conversion system using speech feature extraction (e.g., MFCC or spectrograms) and a neural network or traditional signal processing techniques to map the voice features of one speaker to another.

In this Voice Conversion project, we first extract MFCC features from both the source and target speaker's audio. Then, a Linear Regression model is trained to map the source speaker's features to the target speaker's features. The system then applies this learned mapping to convert the voice features of the source speaker into the target speaker's voice features.

For more advanced voice conversion, methods like CycleGAN or Autoencoders can be used to perform non-linear transformations between speakers.
"""

import numpy as np
import librosa
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
 
# 1. Load audio and extract MFCC features
def extract_mfcc(file_path):
    audio, sr = librosa.load(file_path, sr=None)  # Load the audio file
    mfcc = librosa.feature.mfcc(audio, sr=sr, n_mfcc=13)  # Extract MFCC features
    return np.mean(mfcc, axis=1)  # Use the mean of the MFCC features for conversion
 
# 2. Train the voice conversion model (mapping from source to target speaker)
def train_voice_conversion_model(source_audio, target_audio):
    source_mfcc = extract_mfcc(source_audio)  # Extract MFCC from source speaker
    target_mfcc = extract_mfcc(target_audio)  # Extract MFCC from target speaker
 
    # Linear regression model to map source features to target features
    model = LinearRegression()
    model.fit(source_mfcc.reshape(-1, 1), target_mfcc.reshape(-1, 1))  # Train on MFCC features
 
    return model
 
# 3. Convert source voice to target voice using the trained model
def convert_voice(model, source_audio):
    source_mfcc = extract_mfcc(source_audio)  # Extract MFCC from the source speaker
    converted_mfcc = model.predict(source_mfcc.reshape(-1, 1))  # Map to target speaker's MFCC
    return converted_mfcc
 
# 4. Visualize the original and converted MFCC features
def plot_conversion(source_audio, target_audio, converted_mfcc):
    source_mfcc = extract_mfcc(source_audio)
    target_mfcc = extract_mfcc(target_audio)
 
    plt.figure(figsize=(10, 6))
    plt.subplot(3, 1, 1)
    plt.plot(source_mfcc, label="Source Speaker MFCC")
    plt.title("Source Speaker MFCC")
    plt.subplot(3, 1, 2)
    plt.plot(target_mfcc, label="Target Speaker MFCC")
    plt.title("Target Speaker MFCC")
    plt.subplot(3, 1, 3)
    plt.plot(converted_mfcc, label="Converted MFCC")
    plt.title("Converted Speaker MFCC")
    plt.tight_layout()
    plt.show()
 
# 5. Example usage
source_audio = "path_to_source_speaker_audio.wav"  # Replace with source speaker's audio file
target_audio = "path_to_target_speaker_audio.wav"  # Replace with target speaker's audio file
 
# Train the voice conversion model
model = train_voice_conversion_model(source_audio, target_audio)
 
# Convert the source speaker's voice to the target speaker's voice
converted_mfcc = convert_voice(model, source_audio)
 
# Visualize the MFCC features
plot_conversion(source_audio, target_audio, converted_mfcc)