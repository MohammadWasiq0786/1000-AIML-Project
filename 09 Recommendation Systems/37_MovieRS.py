"""
Project 357. Movie recommendation system
Description:
A movie recommendation system suggests movies to users based on:

User interests (e.g., preferred genres, actors, directors)

Movie features (e.g., genre, cast, plot description)

In this project, we’ll build a movie recommendation system using content-based filtering, where we recommend movies based on their features and the user’s preferences.

About:
✅ What It Does:
Uses TF-IDF to convert movie features (e.g., genre, plot) and the user profile into numerical features

Computes cosine similarity to measure how similar the user’s preferences are to each movie’s features

Recommends top movies based on content similarity between the user’s profile and movie features
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
 
# 1. Simulate movie listings and their features (e.g., genre, plot, actors)
movies = ['The Matrix', 'Inception', 'The Dark Knight', 'The Social Network', 'Forrest Gump']
movie_features = [
    "A hacker discovers a dystopian world where machines control humanity, featuring cyberpunk themes.",
    "A mind-bending thriller about a thief who enters dreams to steal secrets, exploring dreams and reality.",
    "A billionaire playboy turns into a vigilante hero in Gotham City, fighting crime and corruption.",
    "The story of Facebook's creation, focusing on the social and legal challenges behind its success.",
    "A simple man with a kind heart tells the story of his extraordinary life, from the Vietnam War to running a shrimp company."
]
 
# 2. Simulate user preferences (e.g., favorite genres, actors, themes)
user_profile = "I love science fiction and action movies with strong plot twists."
 
# 3. Use TF-IDF to convert movie features and user profile into numerical features
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(movie_features + [user_profile])  # Combine movie features and user profile
 
# 4. Function to recommend movies based on user preferences
def movie_recommendation(user_profile, movies, tfidf_matrix, top_n=3):
    # Compute the cosine similarity between the user profile and movie features
    cosine_sim = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1]).flatten()
    
    # Get the indices of the most similar movies
    similar_indices = cosine_sim.argsort()[-top_n:][::-1]
    
    recommended_movies = [movies[i] for i in similar_indices]
    return recommended_movies
 
# 5. Recommend movies based on the user profile
recommended_movies = movie_recommendation(user_profile, movies, tfidf_matrix)
print(f"Movie Recommendations based on your profile: {recommended_movies}")