"""
Project 358. Music recommendation system
Description:
A music recommendation system suggests songs or albums to users based on:

User preferences (e.g., genre, artist, mood)

Music features (e.g., genre, tempo, lyrics)

In this project, we’ll build a music recommendation system using content-based filtering, where we recommend music based on the features of the music and the user’s preferences.

About:
✅ What It Does:
Uses TF-IDF to convert music features (e.g., genre, lyrics) and the user profile into numerical features

Computes cosine similarity to measure how similar the user’s preferences are to each music track

Recommends top music tracks based on content similarity between the user’s profile and music features
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
 
# 1. Simulate music tracks and their features (e.g., genre, mood, artist)
tracks = ['Song1', 'Song2', 'Song3', 'Song4', 'Song5']
track_features = [
    "Upbeat pop song with catchy lyrics, perfect for a workout playlist.",
    "Slow ballad with emotional lyrics, ideal for a romantic evening.",
    "Fast-paced rock song with heavy guitar riffs, great for an adrenaline boost.",
    "Electronic dance music with high tempo, ideal for a clubbing night.",
    "Classical piano piece, calm and soothing for relaxation or studying."
]
 
# 2. Simulate user preferences (e.g., favorite genres, artists, or mood)
user_profile = "I enjoy upbeat songs with catchy lyrics and electronic music."
 
# 3. Use TF-IDF to convert track features and user profile into numerical features
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(track_features + [user_profile])  # Combine track features and user profile
 
# 4. Function to recommend music based on user preferences
def music_recommendation(user_profile, tracks, tfidf_matrix, top_n=3):
    # Compute the cosine similarity between the user profile and track features
    cosine_sim = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1]).flatten()
    
    # Get the indices of the most similar tracks
    similar_indices = cosine_sim.argsort()[-top_n:][::-1]
    
    recommended_tracks = [tracks[i] for i in similar_indices]
    return recommended_tracks
 
# 5. Recommend music tracks based on the user profile
recommended_tracks = music_recommendation(user_profile, tracks, tfidf_matrix)
print(f"Music Recommendations based on your profile: {recommended_tracks}")