"""
Project 709: Cover Song Identification
Description:
Cover song identification is the task of identifying if a song has been covered by another artist. The challenge lies in identifying the same song despite changes in performance style, instrumentation, and arrangement. In this project, we will implement a cover song identification system using audio feature extraction and a machine learning model (e.g., SVM or KNN) to identify cover songs based on similarity of the audio content.

In this project, we will extract audio features (such as MFCC, chroma, and spectral contrast) from the original and cover versions of the songs. Then, we will train a machine learning model (e.g., Support Vector Machine or K-Nearest Neighbors) to classify whether two songs are a cover version of each other.

Required Libraries:
pip install librosa scikit-learn numpy matplotlib

Explanation:
Audio Feature Extraction: We extract MFCC, chroma, and spectral contrast features from both the original and cover versions of songs using Librosa.

Model Training: We use a Support Vector Classifier (SVC) to classify the songs as either original or cover. We train the model on a dataset of labeled original and cover songs.

Cover Song Matching: The model predicts whether two songs are cover versions of each other based on the extracted audio features. We use the difference between features from two songs to perform the matching.

For a more advanced solution, deep learning models such as Siamese Networks can be used for comparing two audio signals directly and can provide more accurate results.
"""

import librosa
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os
 
# 1. Extract features (MFCC, chroma, spectral contrast) from an audio file
def extract_audio_features(file_path):
    audio, sr = librosa.load(file_path, sr=None)
    
    # Extract MFCC features
    mfcc = librosa.feature.mfcc(audio, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc, axis=1)
    
    # Extract chroma features
    chroma = librosa.feature.chroma_stft(audio, sr=sr)
    chroma_mean = np.mean(chroma, axis=1)
    
    # Extract spectral contrast features
    spectral_contrast = librosa.feature.spectral_contrast(audio, sr=sr)
    spectral_contrast_mean = np.mean(spectral_contrast, axis=1)
    
    # Combine features
    features = np.hstack((mfcc_mean, chroma_mean, spectral_contrast_mean))
    
    return features
 
# 2. Collect the dataset (folder with subfolders named by song titles, and labels for original/cover)
def collect_data(directory):
    X = []  # Features
    y = []  # Labels (1 for cover, 0 for original)
    songs = os.listdir(directory)  # List of song folders
    
    for song in songs:
        song_folder = os.path.join(directory, song)
        if os.path.isdir(song_folder):
            for file in os.listdir(song_folder):
                file_path = os.path.join(song_folder, file)
                if file.endswith('.wav'):  # Assuming all audio files are .wav format
                    features = extract_audio_features(file_path)
                    X.append(features)
                    if 'cover' in song:
                        y.append(1)  # Label 1 for cover songs
                    else:
                        y.append(0)  # Label 0 for original songs
    
    return np.array(X), np.array(y)
 
# 3. Train the cover song identification model
def train_cover_song_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = SVC(kernel='linear')  # Support Vector Classifier for cover song identification
    model.fit(X_train, y_train)  # Train the model
    
    # Evaluate the model
    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    
    return model
 
# 4. Test the model with a new pair of audio files (query the model with two songs)
def classify_cover_song(model, song1_path, song2_path):
    song1_features = extract_audio_features(song1_path)
    song2_features = extract_audio_features(song2_path)
    
    # Combine the features of the two songs to compare them
    features = np.abs(song1_features - song2_features)  # Feature difference
    
    # Predict if the songs are a match (cover or not)
    prediction = model.predict([features])
    if prediction == 1:
        print("The songs are a cover version of each other!")
    else:
        print("The songs are not a cover version of each other.")
 
# 5. Example usage
directory = "path_to_song_dataset"  # Replace with the path to your song dataset folder
X, y = collect_data(directory)  # Collect features and labels from the dataset
model = train_cover_song_model(X, y)  # Train the model
 
# Test the model with a pair of audio files (query two songs for cover song identification)
song1 = "path_to_original_song.wav"  # Replace with path to an original song
song2 = "path_to_cover_song.wav"  # Replace with path to a cover version of the song
classify_cover_song(model, song1, song2)