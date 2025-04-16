"""
Project 93. Speaker recognition system
Description:
A speaker recognition system identifies a person based on their unique voice characteristics. In this project, we use MFCC (Mel-frequency cepstral coefficients) extracted from audio files and build a machine learning model (like SVM or Random Forest) to classify speakers.

About:
🗣️ What This Project Demonstrates:
Extracts voice features (MFCCs) to distinguish speakers

Trains a machine learning model for classification

Visualizes results using a confusion matrix
"""

# Install if not already: pip install librosa scikit-learn matplotlib numpy
 
import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import seaborn as sns
 
# Feature extraction using MFCC
def extract_mfcc(file_path, n_mfcc=13):
    y, sr = librosa.load(file_path, duration=10)  # Use 10 seconds for consistency
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return np.mean(mfccs.T, axis=0)  # Mean across time
 
# Simulated dataset directory structure:
# speakers/alice/audio1.wav, speakers/bob/audio2.wav ...
dataset_path = "speakers"
speakers = os.listdir(dataset_path)
 
features = []
labels = []
 
for speaker in speakers:
    speaker_dir = os.path.join(dataset_path, speaker)
    for file in os.listdir(speaker_dir):
        if file.endswith(".wav"):
            file_path = os.path.join(speaker_dir, file)
            try:
                mfcc = extract_mfcc(file_path)
                features.append(mfcc)
                labels.append(speaker)
            except Exception as e:
                print(f"Error with {file_path}: {e}")
 
# Prepare data
X = np.array(features)
y = np.array(labels)
 
# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
 
# Train speaker classifier
clf = SVC(kernel='linear', probability=True)
clf.fit(X_train, y_train)
 
# Evaluate
y_pred = clf.predict(X_test)
print("Classification Report:\n")
print(classification_report(y_test, y_pred))
 
# Confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=speakers)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=speakers, yticklabels=speakers, cmap='Blues')
plt.title("Speaker Recognition - Confusion Matrix")
plt.xlabel("Predicted Speaker")
plt.ylabel("True Speaker")
plt.tight_layout()
plt.show()