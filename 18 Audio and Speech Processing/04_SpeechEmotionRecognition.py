"""
Project 684: Speech Emotion Recognition
Description:
Speech emotion recognition involves detecting emotional states (such as happiness, sadness, anger, etc.) from speech signals. This is important for applications like affective computing, virtual assistants, and customer service automation. In this project, we will implement a speech emotion recognition system using MFCC (Mel-frequency cepstral coefficients) for feature extraction and a machine learning model (such as SVM or Random Forest) to classify emotions based on the features extracted from the audio.
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
 
# 2. Collect emotion dataset (folder with subfolders named by emotions)
def collect_data(directory):
    X = []  # Features (MFCCs)
    y = []  # Labels (Emotions)
    emotions = os.listdir(directory)  # List of emotion folders
    
    for emotion in emotions:
        emotion_folder = os.path.join(directory, emotion)
        if os.path.isdir(emotion_folder):
            for file in os.listdir(emotion_folder):
                file_path = os.path.join(emotion_folder, file)
                if file.endswith('.wav'):  # Assuming all audio files are .wav format
                    mfcc_features = extract_mfcc(file_path)
                    X.append(mfcc_features)
                    y.append(emotion)  # Label is the emotion
    return np.array(X), np.array(y)
 
# 3. Train the emotion recognition model
def train_emotion_recognition_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = SVC(kernel='linear')  # Support Vector Classifier for emotion recognition
    model.fit(X_train, y_train)  # Train the model
    
    # Evaluate the model
    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    
    return model
 
# 4. Test emotion recognition with a new audio file
def predict_emotion(model, file_path):
    mfcc_features = extract_mfcc(file_path)  # Extract MFCC features from the input file
    predicted_emotion = model.predict([mfcc_features])
    print(f"The emotion is: {predicted_emotion[0]}")
 
# 5. Example usage
directory = "path_to_your_emotion_dataset"  # Replace with the path to your emotion dataset folder
X, y = collect_data(directory)  # Collect features and labels from the dataset
model = train_emotion_recognition_model(X, y)  # Train the model
 
# Test the model with a new audio file
test_file = "path_to_test_audio.wav"  # Replace with a test audio file
predict_emotion(model, test_file)