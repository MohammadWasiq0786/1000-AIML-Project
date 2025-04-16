"""
Project 359. Product recommendation system
Description:
A product recommendation system suggests products to users based on:

User preferences (e.g., product type, budget, brand)

Product features (e.g., category, price, brand)

In this project, we’ll build a product recommendation system using content-based filtering, where we recommend products based on the features of the products and the user’s preferences.

About:
✅ What It Does:
Uses TF-IDF to convert product features (e.g., category, price, brand) and the user profile into numerical features

Computes cosine similarity to measure how similar the user’s preferences are to each product

Recommends top products based on content similarity between the user’s profile and product features
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
 
# 1. Simulate product listings and their features (e.g., category, price, brand)
products = ['Laptop', 'Smartphone', 'Headphones', 'Tablet', 'Smartwatch']
product_features = [
    "High-performance laptop with fast processor, large storage, and great battery life.",
    "Latest smartphone with cutting-edge camera technology, fast processor, and sleek design.",
    "Noise-cancelling headphones with high-quality sound, perfect for travel or work.",
    "Lightweight tablet with a high-resolution display, ideal for reading and entertainment.",
    "Smartwatch with fitness tracking, heart rate monitor, and customizable watch faces."
]
 
# 2. Simulate user preferences (e.g., favorite categories, brand, price range)
user_profile = "I need a fast laptop with a large screen and long battery life."
 
# 3. Use TF-IDF to convert product features and user profile into numerical features
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(product_features + [user_profile])  # Combine product features and user profile
 
# 4. Function to recommend products based on user preferences
def product_recommendation(user_profile, products, tfidf_matrix, top_n=3):
    # Compute the cosine similarity between the user profile and product features
    cosine_sim = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1]).flatten()
    
    # Get the indices of the most similar products
    similar_indices = cosine_sim.argsort()[-top_n:][::-1]
    
    recommended_products = [products[i] for i in similar_indices]
    return recommended_products
 
# 5. Recommend products based on the user profile
recommended_products = product_recommendation(user_profile, products, tfidf_matrix)
print(f"Product Recommendations based on your profile: {recommended_products}")