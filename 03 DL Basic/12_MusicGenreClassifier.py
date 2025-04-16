"""
Project 92. Music genre classifier
Description:
A music genre classifier uses features extracted from audio files (such as tempo, pitch, spectral features) to classify songs into genres like rock, classical, jazz, pop, etc. In this project, we extract features using LibROSA and train a Random Forest classifier to predict genres using the GTZAN dataset or similar audio datasets.

About:
🎵 What This Project Demonstrates:
Extracts meaningful audio features using LibROSA

Trains a classifier for multi-class genre prediction

Evaluates with accuracy and confusion matrix
"""

# Install if not already: pip install librosa scikit-learn matplotlib numpy
 
import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
 
# Function to extract features from an audio file
def extract_features(file_path):
    y, sr = librosa.load(file_path, duration=30)  # Load 30s clip
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
 
    # Take mean of each feature across time
    return np.hstack([
        np.mean(mfccs, axis=1),
        np.mean(chroma, axis=1),
        np.mean(spectral_contrast, axis=1)
    ])
 
# Simulated dataset loading (replace with actual paths in practice)
# For example, folder structure: /genres/blues/xxx.wav, /genres/jazz/yyy.wav etc.
dataset_path = 'genres'  # GTZAN-like structure
genres = os.listdir(dataset_path)
 
features = []
labels = []
 
for genre in genres:
    genre_path = os.path.join(dataset_path, genre)
    for filename in os.listdir(genre_path):
        if filename.endswith(".wav"):
            file_path = os.path.join(genre_path, filename)
            try:
                feat = extract_features(file_path)
                features.append(feat)
                labels.append(genre)
            except Exception as e:
                print(f"Could not process {file_path}: {e}")
 
# Prepare data
X = np.array(features)
y = np.array(labels)
 
# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
 
# Train classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
 
# Predict and evaluate
y_pred = model.predict(X_test)
print("Classification Report:\n")
print(classification_report(y_test, y_pred))
 
# Confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=genres)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, xticklabels=genres, yticklabels=genres, annot=True, fmt='d', cmap='coolwarm')
plt.title("Music Genre Classification - Confusion Matrix")
plt.xlabel("Predicted Genre")
plt.ylabel("True Genre")
plt.tight_layout()
plt.show()