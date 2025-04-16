"""
Project 360. Friend recommendation system
Description:
A friend recommendation system suggests friends to users based on:

User interests (e.g., hobbies, activities, social connections)

Social activity (e.g., mutual friends, shared activities)

In this project, we’ll build a friend recommendation system using content-based filtering, where we recommend potential friends based on shared interests and activities.

About:
✅ What It Does:
Uses TF-IDF to convert user interests into numerical features

Computes cosine similarity to measure how similar the user’s interests are to those of other users

Recommends top friends based on content similarity between the user’s profile and other users’ profiles
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
 
# 1. Simulate user profiles and their interests (e.g., hobbies, activities)
users = ['User1', 'User2', 'User3', 'User4', 'User5']
user_interests = [
    "I enjoy playing basketball, hiking, and watching movies.",
    "I love reading, cooking, and hiking in the mountains.",
    "I'm passionate about photography, traveling, and video games.",
    "I enjoy cooking, reading, and spending time with friends.",
    "I'm into fitness, basketball, and outdoor adventures."
]
 
# 2. Use TF-IDF to convert user interests into numerical features
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(user_interests)  # Convert user interests into TF-IDF vectors
 
# 3. Function to recommend friends based on interest similarity
def friend_recommendation(user_idx, users, tfidf_matrix, top_n=3):
    # Compute the cosine similarity between the user and all other users' interests
    cosine_sim = cosine_similarity(tfidf_matrix[user_idx], tfidf_matrix).flatten()
    
    # Get the indices of the most similar users (excluding the user themselves)
    similar_indices = cosine_sim.argsort()[-top_n-1:-1][::-1]
    
    recommended_friends = [users[i] for i in similar_indices]
    return recommended_friends
 
# 4. Recommend friends for User1 (index 0)
user_idx = 0  # User1
recommended_friends = friend_recommendation(user_idx, users, tfidf_matrix)
print(f"Friend Recommendations for User1: {recommended_friends}")