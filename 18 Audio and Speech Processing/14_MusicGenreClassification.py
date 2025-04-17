"""
Project 693: Music Genre Classification
Description:
Music genre classification is the task of automatically assigning a genre label (e.g., rock, pop, jazz) to a given music track based on its audio features. This project uses MFCC features for audio analysis, and machine learning techniques (like SVM or Random Forest) to classify music into predefined genres. The goal is to analyze music tracks and categorize them into appropriate genres based on their sound features.

In this Music Genre Classification project, we extract MFCC features from audio files and use an SVM classifier to categorize music into genres such as pop, rock, or classical. You can expand the project by using deeper neural networks or incorporating additional audio features like chroma, spectral contrast, or tonnetz for better genre recognition.
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
 
# 2. Collect music genre dataset (folder with subfolders named by genres)
def collect_data(directory):
    X = []  # Features (MFCCs)
    y = []  # Labels (Genres)
    genres = os.listdir(directory)  # List of genre folders
    
    for genre in genres:
        genre_folder = os.path.join(directory, genre)
        if os.path.isdir(genre_folder):
            for file in os.listdir(genre_folder):
                file_path = os.path.join(genre_folder, file)
                if file.endswith('.wav'):  # Assuming all audio files are .wav format
                    mfcc_features = extract_mfcc(file_path)
                    X.append(mfcc_features)
                    y.append(genre)  # Label is the genre (folder name)
    return np.array(X), np.array(y)
 
# 3. Train the music genre classification model
def train_genre_classification_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = SVC(kernel='linear')  # Support Vector Classifier for music genre classification
    model.fit(X_train, y_train)  # Train the model
    
    # Evaluate the model
    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    
    return model
 
# 4. Test the music genre classification model with a new audio file
def classify_genre(model, file_path):
    mfcc_features = extract_mfcc(file_path)  # Extract MFCC features from the input file
    predicted_genre = model.predict([mfcc_features])
    print(f"The predicted genre is: {predicted_genre[0]}")
 
# 5. Example usage
directory = "path_to_music_genre_dataset"  # Replace with the path to your music genre dataset folder
X, y = collect_data(directory)  # Collect features and labels from the dataset
model = train_genre_classification_model(X, y)  # Train the model
 
# Test the model with a new audio file
test_file = "path_to_test_music.wav"  # Replace with a test music file
classify_genre(model, test_file)