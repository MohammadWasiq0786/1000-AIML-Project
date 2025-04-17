"""
Project 682: Speaker Identification
Description:
Speaker identification involves determining who is speaking based on their voice characteristics. It is used in applications like voice biometrics, user authentication, and forensics. In this project, we will implement a speaker identification system that can recognize different speakers from a set of recorded speech data. We'll use MFCC (Mel-frequency cepstral coefficients) for feature extraction and a machine learning model (e.g., KNN or SVM) to classify the speakers based on their voice features.
"""

import os
import numpy as np
import librosa
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
 
# 1. Load audio files and extract MFCC features
def extract_mfcc(file_path):
    audio, sr = librosa.load(file_path, sr=None)  # Load the audio file
    mfcc = librosa.feature.mfcc(audio, sr=sr, n_mfcc=13)  # Extract MFCC features
    return np.mean(mfcc, axis=1)  # Use the mean MFCC features for classification
 
# 2. Collect dataset (folder with subfolders named by speaker names)
def collect_data(directory):
    X = []  # Features
    y = []  # Labels
    speakers = os.listdir(directory)  # List of speaker folders
    
    for speaker in speakers:
        speaker_folder = os.path.join(directory, speaker)
        if os.path.isdir(speaker_folder):
            for file in os.listdir(speaker_folder):
                file_path = os.path.join(speaker_folder, file)
                if file.endswith('.wav'):  # Assuming all audio files are .wav format
                    mfcc_features = extract_mfcc(file_path)
                    X.append(mfcc_features)
                    y.append(speaker)  # Label is the folder name (speaker)
    
    return np.array(X), np.array(y)
 
# 3. Train a KNN classifier for speaker identification
def train_speaker_recognition_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = KNeighborsClassifier(n_neighbors=3)  # Initialize KNN classifier
    model.fit(X_train, y_train)  # Train the model
    
    # Evaluate the model
    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    
    return model
 
# 4. Test speaker identification with a new audio file
def identify_speaker(model, file_path):
    mfcc_features = extract_mfcc(file_path)  # Extract MFCC features from the input file
    predicted_speaker = model.predict([mfcc_features])
    print(f"The speaker is: {predicted_speaker[0]}")
 
# 5. Example usage
directory = "path_to_your_speaker_dataset"  # Replace with the path to your dataset folder
X, y = collect_data(directory)  # Collect features and labels from the dataset
model = train_speaker_recognition_model(X, y)  # Train the model
 
# Test the model with a new audio file
test_file = "path_to_test_audio.wav"  # Replace with a test audio file
identify_speaker(model, test_file)