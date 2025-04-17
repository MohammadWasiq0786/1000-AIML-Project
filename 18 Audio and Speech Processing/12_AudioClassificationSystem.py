"""
Project 691: Audio Classification System
Description:
An audio classification system automatically assigns labels to audio data based on its content. This is useful for applications like genre classification, emotion detection, and speech recognition. In this project, we will build a simple audio classification system using MFCC (Mel-frequency cepstral coefficients) features and a machine learning model such as SVM or Random Forest to classify different audio types. We will use a dataset of labeled audio files for training and testing.

In this Audio Classification System, we use MFCC features extracted from audio files to train a Support Vector Classifier (SVC). The model is then used to classify new audio samples based on the learned features.
"""

import numpy as np
import librosa
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os
 
# 1. Extract MFCC features from an audio file
def extract_mfcc(file_path):
    audio, sr = librosa.load(file_path, sr=None)  # Load the audio file
    mfcc = librosa.feature.mfcc(audio, sr=sr, n_mfcc=13)  # Extract MFCC features
    return np.mean(mfcc, axis=1)  # Use the mean of the MFCC features for classification
 
# 2. Collect audio dataset (folder with subfolders named by class labels)
def collect_data(directory):
    X = []  # Features (MFCCs)
    y = []  # Labels (Audio class labels)
    classes = os.listdir(directory)  # List of class folders
    
    for audio_class in classes:
        class_folder = os.path.join(directory, audio_class)
        if os.path.isdir(class_folder):
            for file in os.listdir(class_folder):
                file_path = os.path.join(class_folder, file)
                if file.endswith('.wav'):  # Assuming all audio files are .wav format
                    mfcc_features = extract_mfcc(file_path)
                    X.append(mfcc_features)
                    y.append(audio_class)  # Label is the class folder name
    return np.array(X), np.array(y)
 
# 3. Train the audio classification model
def train_audio_classification_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = SVC(kernel='linear')  # Support Vector Classifier for audio classification
    model.fit(X_train, y_train)  # Train the model
    
    # Evaluate the model
    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    
    return model
 
# 4. Test the audio classification model with a new audio file
def classify_audio(model, file_path):
    mfcc_features = extract_mfcc(file_path)  # Extract MFCC features from the input file
    predicted_class = model.predict([mfcc_features])
    print(f"The audio class is: {predicted_class[0]}")
 
# 5. Example usage
directory = "path_to_audio_dataset"  # Replace with the path to your audio dataset folder
X, y = collect_data(directory)  # Collect features and labels from the dataset
model = train_audio_classification_model(X, y)  # Train the model
 
# Test the model with a new audio file
test_file = "path_to_test_audio.wav"  # Replace with a test audio file
classify_audio(model, test_file)