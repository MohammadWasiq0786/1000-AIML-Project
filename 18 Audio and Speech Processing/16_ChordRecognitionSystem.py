"""
Project 695: Chord Recognition System
Description:
Chord recognition involves detecting musical chords from an audio signal. This is essential for applications like music transcription, music analysis, and automatic accompaniment systems. In this project, we will implement a chord recognition system that analyzes an audio file, extracts relevant features (such as MFCC, chromagram, or spectrogram), and classifies them into musical chord types (e.g., C major, A minor). We will use a machine learning model like SVM or Random Forest to perform chord classification.

In this Chord Recognition System, we use chromagram features, which represent the energy content in each pitch class, to detect musical chords in audio signals. A Support Vector Classifier (SVC) is used to classify the chords. This system can be extended to include more advanced features or neural networks for more complex chord recognition tasks.
"""

import numpy as np
import librosa
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os
 
# 1. Extract chromagram from an audio file for chord recognition
def extract_chromagram(file_path):
    audio, sr = librosa.load(file_path, sr=None)  # Load the audio file
    chroma = librosa.feature.chroma_stft(audio, sr=sr)  # Extract chroma features
    return np.mean(chroma, axis=1)  # Use the mean chroma values for classification
 
# 2. Collect chord dataset (folder with subfolders named by chord types)
def collect_data(directory):
    X = []  # Features (Chroma)
    y = []  # Labels (Chord types)
    chords = os.listdir(directory)  # List of chord folders
    
    for chord in chords:
        chord_folder = os.path.join(directory, chord)
        if os.path.isdir(chord_folder):
            for file in os.listdir(chord_folder):
                file_path = os.path.join(chord_folder, file)
                if file.endswith('.wav'):  # Assuming all audio files are .wav format
                    chroma_features = extract_chromagram(file_path)
                    X.append(chroma_features)
                    y.append(chord)  # Label is the chord type (folder name)
    return np.array(X), np.array(y)
 
# 3. Train the chord recognition model
def train_chord_classification_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = SVC(kernel='linear')  # Support Vector Classifier for chord classification
    model.fit(X_train, y_train)  # Train the model
    
    # Evaluate the model
    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    
    return model
 
# 4. Test the chord recognition model with a new audio file
def classify_chord(model, file_path):
    chroma_features = extract_chromagram(file_path)  # Extract chroma features from the input file
    predicted_chord = model.predict([chroma_features])
    print(f"The predicted chord is: {predicted_chord[0]}")
 
# 5. Example usage
directory = "path_to_chord_dataset"  # Replace with the path to your chord dataset folder
X, y = collect_data(directory)  # Collect features and labels from the dataset
model = train_chord_classification_model(X, y)  # Train the model
 
# Test the model with a new audio file
test_file = "path_to_test_chord_audio.wav"  # Replace with a test chord audio file
classify_chord(model, test_file)