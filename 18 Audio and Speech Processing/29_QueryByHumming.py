"""
Project 708: Query by Humming Implementation
Description:
Query by humming allows users to search for a song or melody by humming a portion of the song. This is a common feature in music recognition applications like Shazam or SoundHound. In this project, we will implement a basic query by humming system where the system extracts melody features from the hummed audio and compares them with stored song melodies to identify the song. The system will use dynamic time warping (DTW) or nearest neighbor search for matching.

To implement the Query by Humming system, we will first extract the melody features from both the hummed audio and the reference songs. We will use Dynamic Time Warping (DTW) to compare the two audio signals and find the closest match.

Install Required Libraries:
pip install librosa scipy numpy matplotlib

Explanation:
Audio Features Extraction: We use MFCC to extract melody features from the hummed query and reference songs. MFCC captures the spectral properties of the audio, which is useful for matching melodies.

Dynamic Time Warping (DTW): We use DTW to compare the sequence of features from the hummed query with the reference songs. DTW finds the optimal alignment between the two signals by calculating the distance matrix.

Matching: The system computes the DTW distance between the hummed query and each reference song. The song with the smallest distance is considered the best match.

This system is a simplified version. For better results, advanced models like Siamese networks or neural networks trained on large datasets of hummed queries and reference songs can be used.
"""

import librosa
import numpy as np
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
 
# 1. Load the audio file
def load_audio(file_path):
    audio, sr = librosa.load(file_path, sr=None)
    return audio, sr
 
# 2. Extract melody features (MFCC) from the hummed query or reference song
def extract_melody_features(audio, sr, n_mfcc=13):
    # Extract MFCC features
    mfcc = librosa.feature.mfcc(audio, sr=sr, n_mfcc=n_mfcc)
    return np.mean(mfcc, axis=1)
 
# 3. Dynamic Time Warping (DTW) for comparing two audio signals
def dynamic_time_warping(query_features, reference_features):
    # Compute the DTW distance between the query and reference features
    distance_matrix = cdist(query_features.reshape(-1, 1), reference_features.reshape(-1, 1), metric='euclidean')
    return np.sum(distance_matrix)
 
# 4. Match the hummed query to reference songs in the database
def query_by_humming(query_audio, reference_audios):
    # Extract features from the query
    query_features = extract_melody_features(query_audio, sr)
    
    # Compare the query with each reference audio in the database
    distances = []
    for ref_audio in reference_audios:
        ref_features = extract_melody_features(ref_audio, sr)
        distance = dynamic_time_warping(query_features, ref_features)
        distances.append(distance)
    
    # Find the best match (minimum distance)
    best_match_index = np.argmin(distances)
    return best_match_index, distances[best_match_index]
 
# 5. Example usage
query_audio_file = "path_to_query_humming.wav"  # Replace with the path to the hummed query
reference_audio_files = ["path_to_reference_song1.wav", "path_to_reference_song2.wav"]  # Replace with reference songs
 
# Load the query humming and reference songs
query_audio, sr = load_audio(query_audio_file)
reference_audios = [load_audio(file)[0] for file in reference_audio_files]
 
# Find the best match for the hummed query
best_match_index, distance = query_by_humming(query_audio, reference_audios)
 
# Print the best match result
print(f"The best match is reference song {best_match_index + 1} with a distance of {distance}")