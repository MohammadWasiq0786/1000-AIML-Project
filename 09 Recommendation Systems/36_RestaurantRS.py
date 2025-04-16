"""
Project 356. Restaurant recommendation system
Description:
A restaurant recommendation system suggests restaurants to users based on:

User preferences (e.g., cuisine type, budget, location)

Restaurant features (e.g., cuisine, ambiance, location)

In this project, we’ll build a restaurant recommendation system using content-based filtering, where we recommend restaurants based on the features of the restaurant and the user’s preferences.

About:
✅ What It Does:
Uses TF-IDF to convert restaurant features and the user profile into numerical features

Computes cosine similarity to measure how similar the user’s dining preferences are to each restaurant

Recommends top restaurants based on content similarity between the user’s profile and restaurant features
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
 
# 1. Simulate restaurant listings and their features (e.g., cuisine, ambiance, price range)
restaurants = ['Italian Bistro', 'Sushi Place', 'Taco Shop', 'French Cafe', 'Indian Spice']
restaurant_features = [
    "Cozy Italian bistro serving classic pasta, pizza, and wine in a romantic atmosphere.",
    "Traditional Japanese sushi and sashimi served in a modern, minimalistic ambiance.",
    "Casual taco shop offering Mexican street food with a vibrant, colorful setting.",
    "Charming French cafe serving croissants, pastries, and gourmet coffee in an intimate environment.",
    "Authentic Indian restaurant offering a variety of curries, tandoori dishes, and aromatic spices."
]
 
# 2. Simulate user preferences (e.g., cuisine, dining style)
user_profile = "I love Italian food and casual dining experiences with a vibrant atmosphere."
 
# 3. Use TF-IDF to convert restaurant features and user profile into numerical features
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(restaurant_features + [user_profile])  # Combine restaurant features and user profile
 
# 4. Function to recommend restaurants based on user preferences
def restaurant_recommendation(user_profile, restaurants, tfidf_matrix, top_n=3):
    # Compute the cosine similarity between the user profile and restaurant features
    cosine_sim = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1]).flatten()
    
    # Get the indices of the most similar restaurants
    similar_indices = cosine_sim.argsort()[-top_n:][::-1]
    
    recommended_restaurants = [restaurants[i] for i in similar_indices]
    return recommended_restaurants
 
# 5. Recommend restaurants based on the user profile
recommended_restaurants = restaurant_recommendation(user_profile, restaurants, tfidf_matrix)
print(f"Restaurant Recommendations based on your profile: {recommended_restaurants}")