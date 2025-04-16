"""
Project 351. Audio recommendation system
Description:
An audio recommendation system uses audio features to recommend items such as:

Music tracks based on similarity (e.g., genre, mood)

Podcasts or audiobooks based on the user's listening preferences

In this project, we’ll create an audio recommendation system by extracting audio features using pre-trained models or feature extraction techniques (like MFCC — Mel Frequency Cepstral Coefficients) and recommend audio content based on these features.

About:
✅ What It Does:
Extracts audio features (MFCC) from audio files (e.g., music tracks)

Cosine similarity is used to compare audio features between the query item and all other items

Provides audio recommendations based on similarity of audio characteristics like pitch, timbre, and rhythm
"""

import numpy as np
import pandas as pd
import librosa
from sklearn.metrics.pairwise import cosine_similarity
 
# 1. Simulate audio items (e.g., music tracks) with their file paths
items = ['Song1', 'Song2', 'Song3', 'Song4', 'Song5']
audio_paths = {
    'Song1': 'path_to_song1.wav',
    'Song2': 'path_to_song2.wav',
    'Song3': 'path_to_song3.wav',
    'Song4': 'path_to_song4.wav',
    'Song5': 'path_to_song5.wav'
}
 
# 2. Function to extract audio features (MFCC - Mel Frequency Cepstral Coefficients)
def extract_audio_features(audio_path):
    # Load audio file
    y, sr = librosa.load(audio_path, sr=None)  # y = audio signal, sr = sample rate
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)  # Extract MFCC features
    mfcc = np.mean(mfcc, axis=1)  # Take mean of MFCC coefficients across time frames
    return mfcc
 
# 3. Function to recommend audio items based on audio feature similarity
def audio_recommendation(query_item, audio_paths, top_n=3):
    # Extract audio features for the query item
    query_features = extract_audio_features(audio_paths[query_item])
    
    similarities = []
    for item, path in audio_paths.items():
        if item != query_item:  # Skip the query item itself
            item_features = extract_audio_features(path)
            similarity = cosine_similarity([query_features], [item_features])
            similarities.append((item, similarity[0][0]))
    
    # Sort by similarity score (highest first)
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    # Return the top N recommended items based on audio similarity
    return [item for item, _ in similarities[:top_n]]
 
# 4. Recommend audio items based on the audio similarity of 'Song1'
recommended_items = audio_recommendation('Song1', audio_paths)
print(f"Audio Recommendations for Song1: {recommended_items}")