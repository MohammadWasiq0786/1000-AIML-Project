"""
Project 692: Environmental Sound Classification
Description:
Environmental sound classification involves categorizing various types of sounds in the environment (such as traffic noise, birds chirping, or people talking). It is important for applications like smart home systems, urban monitoring, and wildlife observation. In this project, we will implement an environmental sound classification system using MFCC (Mel-frequency cepstral coefficients) for feature extraction and a machine learning classifier (such as Random Forest or SVM) to classify different environmental sounds.

In this Environmental Sound Classification project, we use MFCC features to represent environmental sounds, and a Support Vector Classifier (SVC) to categorize these sounds into different classes (e.g., traffic noise, rain, birds). You can expand this project by incorporating more advanced models like Convolutional Neural Networks (CNNs) for better accuracy in sound classification.
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
 
# 2. Collect environmental sound dataset (folder with subfolders named by sound types)
def collect_data(directory):
    X = []  # Features (MFCCs)
    y = []  # Labels (Sound types)
    sound_types = os.listdir(directory)  # List of sound type folders
    
    for sound_type in sound_types:
        sound_folder = os.path.join(directory, sound_type)
        if os.path.isdir(sound_folder):
            for file in os.listdir(sound_folder):
                file_path = os.path.join(sound_folder, file)
                if file.endswith('.wav'):  # Assuming all audio files are .wav format
                    mfcc_features = extract_mfcc(file_path)
                    X.append(mfcc_features)
                    y.append(sound_type)  # Label is the sound type (folder name)
    return np.array(X), np.array(y)
 
# 3. Train the environmental sound classification model
def train_sound_classification_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = SVC(kernel='linear')  # Support Vector Classifier for environmental sound classification
    model.fit(X_train, y_train)  # Train the model
    
    # Evaluate the model
    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    
    return model
 
# 4. Test the environmental sound classification model with a new audio file
def classify_sound(model, file_path):
    mfcc_features = extract_mfcc(file_path)  # Extract MFCC features from the input file
    predicted_class = model.predict([mfcc_features])
    print(f"The sound type is: {predicted_class[0]}")
 
# 5. Example usage
directory = "path_to_sound_dataset"  # Replace with the path to your environmental sound dataset folder
X, y = collect_data(directory)  # Collect features and labels from the dataset
model = train_sound_classification_model(X, y)  # Train the model
 
# Test the model with a new audio file
test_file = "path_to_test_audio.wav"  # Replace with a test audio file
classify_sound(model, test_file)

