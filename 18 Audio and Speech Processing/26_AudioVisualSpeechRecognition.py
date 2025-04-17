"""
Project 705: Audio-Visual Speech Recognition
Description:
Audio-Visual Speech Recognition (AVSR) involves using both audio (speech signal) and visual (lip movements or facial expressions) information to improve the accuracy of speech recognition. This is especially useful in noisy environments where the audio alone might not be sufficient for accurate recognition. In this project, we will build an audio-visual speech recognition system that combines both audio features (such as MFCC) and visual features (such as mouth region images) for speech recognition.

We will use a deep learning model to process both the audio and video data. In this simplified implementation, we'll extract audio features using MFCC and visual features by processing the mouth region from video frames using OpenCV. The model will be a simple neural network that takes both types of input to predict the spoken words.

Requirements:
OpenCV: For extracting video frames.

Librosa: For extracting audio features.

TensorFlow/Keras: For building the deep learning model.

First, ensure you have the required libraries:

pip install opencv-python librosa tensorflow

Explanation:
In this Audio-Visual Speech Recognition (AVSR) project, we process both audio and visual inputs:

Audio Features: We use MFCC to extract audio features, which represent the spectral properties of the audio signal.

Visual Features: We extract the mouth region from the video using OpenCV, and use it as a source of visual features for speech recognition.

Deep Learning Model: We build a CNN-based model for visual features and a simple feed-forward network for audio features. These are then combined in a concatenation layer, and the output layer classifies the speech event.

For real training, you would need a large audio-visual dataset (e.g., GRID corpus, AVSpeech, or LRS2) containing both the audio and video of speech for training and validation.
"""

import numpy as np
import librosa
import cv2
import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt
 
# 1. Extract audio features (MFCC)
def extract_audio_features(file_path, n_mfcc=13):
    audio, sr = librosa.load(file_path, sr=None)
    mfcc = librosa.feature.mfcc(audio, sr=sr, n_mfcc=n_mfcc)
    return np.mean(mfcc, axis=1)
 
# 2. Extract visual features (mouth region)
def extract_visual_features(video_file):
    # Open video capture
    cap = cv2.VideoCapture(video_file)
    
    # Load pre-trained face and mouth detectors
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    mouth_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_mcs_mouth.xml')
    
    frames = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert to grayscale for detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces in the frame
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        for (x, y, w, h) in faces:
            # Crop the mouth region (mouth is usually at the lower half of the face)
            roi_gray = gray[y + int(h/2): y + h, x: x + w]
            mouth = mouth_cascade.detectMultiScale(roi_gray, 1.7, 11)
            
            for (mx, my, mw, mh) in mouth:
                mouth_img = frame[y + int(h/2) + my: y + int(h/2) + my + mh, x + mx: x + mx + mw]
                mouth_resized = cv2.resize(mouth_img, (64, 64))  # Resize for input into the model
                frames.append(mouth_resized)
                
    cap.release()
    return np.array(frames)
 
# 3. Build the audio-visual speech recognition model
def build_avsr_model(audio_input_shape, visual_input_shape):
    audio_input = layers.Input(shape=audio_input_shape)
    visual_input = layers.Input(shape=visual_input_shape)
    
    # Audio Model: Simple Dense Layers
    audio_x = layers.Dense(128, activation='relu')(audio_input)
    audio_x = layers.Dense(64, activation='relu')(audio_x)
    
    # Visual Model: Convolutional Layers for Image Processing
    visual_x = layers.Conv2D(32, (3, 3), activation='relu')(visual_input)
    visual_x = layers.MaxPooling2D((2, 2))(visual_x)
    visual_x = layers.Flatten()(visual_x)
    visual_x = layers.Dense(64, activation='relu')(visual_x)
    
    # Concatenate audio and visual features
    combined = layers.concatenate([audio_x, visual_x])
    
    # Final Dense Layer for classification
    output = layers.Dense(10, activation='softmax')(combined)  # Assuming 10 classes for output (words or commands)
    
    model = tf.keras.Model(inputs=[audio_input, visual_input], outputs=output)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model
 
# 4. Example usage
audio_file = 'path_to_audio_file.wav'  # Replace with your audio file path
video_file = 'path_to_video_file.mp4'  # Replace with your video file path
 
# Extract audio features (MFCC)
audio_features = extract_audio_features(audio_file)
 
# Extract visual features (mouth region)
visual_features = extract_visual_features(video_file)
 
# Reshape data for model input
audio_features = np.reshape(audio_features, (1, -1))  # Add batch dimension
visual_features = np.reshape(visual_features, (1, 64, 64, 3))  # Add batch and channel dimension
 
# Build and compile the model
model = build_avsr_model(audio_input_shape=(audio_features.shape[1],), visual_input_shape=(64, 64, 3))
 
# Train the model (use a labeled dataset for real training)
# model.fit([audio_features, visual_features], labels, epochs=10)
 
# Predict the speech event (class) for the input
prediction = model.predict([audio_features, visual_features])
predicted_class = np.argmax(prediction)
print(f"Predicted Speech Event Class: {predicted_class}")