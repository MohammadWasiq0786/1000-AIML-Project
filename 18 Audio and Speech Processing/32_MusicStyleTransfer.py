"""
Project 711: Music Style Transfer
Description:
Music style transfer refers to the process of transferring the style (e.g., tempo, instrument arrangement) of one music piece to another while preserving the content (e.g., melody or harmony). This technique is inspired by image style transfer and can be used to generate new musical compositions by applying the style of one track to the melody of another. In this project, we will implement a music style transfer system using neural networks to modify the style of a song while keeping its melody intact.

For simplicity, we will use feature extraction methods like MFCC and Chroma to represent the content and style of the audio. A neural network model, such as a Convolutional Neural Network (CNN), can be trained to map one piece of music to another, transferring style while preserving the melody.

Required Libraries:
pip install librosa tensorflow numpy

Explanation:
In this Music Style Transfer project:

Feature Extraction: We use MFCC to represent the content of the audio and Chroma features to capture the style of the music.

Neural Network Model: We build a CNN-based model to extract features from both the content and style of the music. The model combines these features and learns how to generate a new audio signal that combines the content from one track and the style from another.

Training and Application: The model is trained with paired audio data (one content and one style track). After training, we can apply the learned style to new content tracks to generate stylized music.

This is a basic approach to music style transfer. More sophisticated techniques like WaveNet or GANs can be used for higher-quality results.
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
 
# 2. Extract audio features (MFCC, Chroma) from the audio
def extract_features(audio, sr, n_mfcc=13):
    # Extract MFCC features (used for content)
    mfcc = librosa.feature.mfcc(audio, sr=sr, n_mfcc=n_mfcc)
    
    # Extract Chroma features (used for style)
    chroma = librosa.feature.chroma_stft(audio, sr=sr)
    
    return mfcc, chroma
 
# 3. Define the style transfer model (CNN-based)
def build_style_transfer_model(content_shape, style_shape):
    content_input = layers.Input(shape=content_shape)
    style_input = layers.Input(shape=style_shape)
 
    # CNN layers for content extraction
    content_x = layers.Conv1D(64, 3, activation='relu')(content_input)
    content_x = layers.MaxPooling1D(2)(content_x)
    content_x = layers.Conv1D(128, 3, activation='relu')(content_x)
    content_x = layers.MaxPooling1D(2)(content_x)
    
    # CNN layers for style extraction
    style_x = layers.Conv1D(64, 3, activation='relu')(style_input)
    style_x = layers.MaxPooling1D(2)(style_x)
    style_x = layers.Conv1D(128, 3, activation='relu')(style_x)
    style_x = layers.MaxPooling1D(2)(style_x)
 
    # Combine content and style features
    combined = layers.concatenate([content_x, style_x])
    
    # Fully connected layers to generate the stylized audio
    x = layers.Flatten()(combined)
    x = layers.Dense(128, activation='relu')(x)
    output = layers.Dense(content_shape[0], activation='linear')(x)
 
    model = tf.keras.Model(inputs=[content_input, style_input], outputs=output)
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model
 
# 4. Train the model for style transfer (using a paired content and style dataset)
def train_style_transfer_model(model, content_audio, style_audio, epochs=10):
    # Extract features from both content and style audio
    content_mfcc, content_chroma = extract_features(content_audio, sr)
    style_mfcc, style_chroma = extract_features(style_audio, sr)
    
    # Train the model
    model.fit([content_mfcc, style_chroma], content_mfcc, epochs=epochs)
    
# 5. Apply style transfer to new content audio
def apply_style_transfer(model, content_audio, style_audio):
    # Extract features from both content and style audio
    content_mfcc, content_chroma = extract_features(content_audio, sr)
    style_mfcc, style_chroma = extract_features(style_audio, sr)
    
    # Generate the stylized audio
    stylized_audio = model.predict([content_mfcc, style_chroma])
    return stylized_audio
 
# 6. Example usage
content_audio_file = 'path_to_content_audio.wav'  # Replace with the content audio path
style_audio_file = 'path_to_style_audio.wav'  # Replace with the style audio path
 
# Load the content and style audio
content_audio, sr = load_audio(content_audio_file)
style_audio, _ = load_audio(style_audio_file)
 
# Build the style transfer model
model = build_style_transfer_model(content_shape=(13, content_audio.shape[0]), style_shape=(13, style_audio.shape[0]))
 
# Train the model (with paired content and style audio)
train_style_transfer_model(model, content_audio, style_audio, epochs=10)
 
# Apply style transfer to new content
stylized_audio = apply_style_transfer(model, content_audio, style_audio)
 
# Plot the original and stylized audio
plt.figure(figsize=(10, 6))
 
# Plot content audio
plt.subplot(2, 1, 1)
plt.plot(content_audio)
plt.title("Original Content Audio")
 
# Plot stylized audio
plt.subplot(2, 1, 2)
plt.plot(stylized_audio[0])
plt.title("Stylized Audio (with Transferred Style)")
 
plt.tight_layout()
plt.show()