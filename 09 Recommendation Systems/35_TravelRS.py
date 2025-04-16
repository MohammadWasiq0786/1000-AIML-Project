"""
Project 355. Travel recommendation system
Description:
A travel recommendation system suggests travel destinations to users based on:

User preferences (e.g., interests, budget, travel style)

Destination features (e.g., climate, culture, activities)

In this project, we’ll create a travel recommendation system using content-based filtering, where we recommend destinations based on their features and the user’s preferences.

About:
✅ What It Does:
Uses TF-IDF to convert destination features and user preferences into numerical features

Computes cosine similarity to measure how similar the user’s interests are to each destination

Recommends top destinations based on content similarity between the user’s profile and destination features
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
 
# 1. Simulate travel destinations and their features (e.g., climate, activities, culture)
destinations = ['Paris', 'New York', 'Tokyo', 'Sydney', 'Barcelona']
destination_features = [
    "Romantic city with iconic landmarks, museums, and fine dining experiences.",
    "Vibrant city with famous landmarks, diverse culture, and bustling nightlife.",
    "A city with cutting-edge technology, rich culture, and unique culinary experiences.",
    "A city with beautiful beaches, iconic landmarks, and a laid-back lifestyle.",
    "A city known for its architecture, history, beaches, and vibrant nightlife."
]
 
# 2. Simulate user preferences (interests, travel style, etc.)
user_profile = "I enjoy cultural experiences, exploring museums, and experiencing unique cuisines."
 
# 3. Use TF-IDF to convert destination features and user profile into numerical features
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(destination_features + [user_profile])  # Combine destination features and user profile
 
# 4. Function to recommend travel destinations based on user preferences
def travel_recommendation(user_profile, destinations, tfidf_matrix, top_n=3):
    # Compute the cosine similarity between the user profile and destination features
    cosine_sim = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1]).flatten()
    
    # Get the indices of the most similar destinations
    similar_indices = cosine_sim.argsort()[-top_n:][::-1]
    
    recommended_destinations = [destinations[i] for i in similar_indices]
    return recommended_destinations
 
# 5. Recommend travel destinations based on the user profile
recommended_destinations = travel_recommendation(user_profile, destinations, tfidf_matrix)
print(f"Travel Recommendations based on your profile: {recommended_destinations}")