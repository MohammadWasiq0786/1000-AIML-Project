"""
Project 694: Musical Instrument Recognition
Description:
Musical instrument recognition involves classifying the type of instrument producing a given sound, such as piano, guitar, or violin. This can be applied in fields like music analysis, automatic tagging of music, and music education. In this project, we will implement a musical instrument recognition system using MFCC features for sound feature extraction and a machine learning classifier (e.g., SVM or Random Forest) to classify different musical instruments from their audio signals.

In this Musical Instrument Recognition project, we use MFCC features to represent the audio characteristics of musical instruments, and a Support Vector Classifier (SVC) to classify them into instrument categories. The system can be extended to include other feature sets like chroma, spectral contrast, or zero-crossing rate for more robust recognition.
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
 
# 2. Collect music instrument dataset (folder with subfolders named by instrument types)
def collect_data(directory):
    X = []  # Features (MFCCs)
    y = []  # Labels (Instrument types)
    instruments = os.listdir(directory)  # List of instrument folders
    
    for instrument in instruments:
        instrument_folder = os.path.join(directory, instrument)
        if os.path.isdir(instrument_folder):
            for file in os.listdir(instrument_folder):
                file_path = os.path.join(instrument_folder, file)
                if file.endswith('.wav'):  # Assuming all audio files are .wav format
                    mfcc_features = extract_mfcc(file_path)
                    X.append(mfcc_features)
                    y.append(instrument)  # Label is the instrument type (folder name)
    return np.array(X), np.array(y)
 
# 3. Train the musical instrument recognition model
def train_instrument_classification_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = SVC(kernel='linear')  # Support Vector Classifier for instrument classification
    model.fit(X_train, y_train)  # Train the model
    
    # Evaluate the model
    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    
    return model
 
# 4. Test the musical instrument recognition model with a new audio file
def classify_instrument(model, file_path):
    mfcc_features = extract_mfcc(file_path)  # Extract MFCC features from the input file
    predicted_instrument = model.predict([mfcc_features])
    print(f"The predicted instrument is: {predicted_instrument[0]}")
 
# 5. Example usage
directory = "path_to_instrument_dataset"  # Replace with the path to your instrument dataset folder
X, y = collect_data(directory)  # Collect features and labels from the dataset
model = train_instrument_classification_model(X, y)  # Train the model
 
# Test the model with a new audio file
test_file = "path_to_test_instrument_audio.wav"  # Replace with a test instrument audio file
classify_instrument(model, test_file)