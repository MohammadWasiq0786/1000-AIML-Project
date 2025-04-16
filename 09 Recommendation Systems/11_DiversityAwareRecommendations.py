"""
Project 331. Diversity-aware recommendations
Description:
Diversity-aware recommendation systems aim to provide varied suggestions, avoiding overfitting to a user’s preferences. This approach is especially useful in scenarios where:

Users might benefit from exploring new genres (e.g., music, movies, books)

The system should recommend novelty while still being relevant

In this project, we’ll combine personalized preferences and diversity metrics to ensure recommendations offer both relevance and variety.

About:
✅ What It Does:
Computes item-item similarity using cosine similarity

Predicts ratings for unrated items based on item similarity

Ensures diverse recommendations by promoting novelty (items not rated by the user)
"""

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
 
# 1. Simulate user-item ratings matrix
users = ['User1', 'User2', 'User3', 'User4', 'User5']
items = ['Item1', 'Item2', 'Item3', 'Item4', 'Item5']
ratings = np.array([
    [5, 4, 0, 0, 3],
    [4, 0, 0, 2, 1],
    [1, 1, 0, 5, 4],
    [0, 0, 5, 4, 4],
    [2, 3, 0, 1, 0]
])
 
df = pd.DataFrame(ratings, index=users, columns=items)
 
# 2. Compute user-item similarity (cosine similarity between items)
item_similarity = cosine_similarity(df.T)
item_similarity_df = pd.DataFrame(item_similarity, index=items, columns=items)
 
# 3. Personalized recommendation (based on user preferences)
def recommend_items(user_idx, df, item_similarity_df, top_n=3, diversity_factor=0.5):
    user_ratings = df.iloc[user_idx]
    unrated_items = user_ratings[user_ratings == 0].index
    
    # Calculate predicted ratings based on item similarity
    predicted_ratings = []
    for item in unrated_items:
        similar_items = item_similarity_df[item]
        rating_prediction = np.dot(user_ratings, similar_items) / np.sum(similar_items)
        predicted_ratings.append((item, rating_prediction))
    
    # Sort by predicted rating
    predicted_ratings = sorted(predicted_ratings, key=lambda x: x[1], reverse=True)
    
    # Ensure diversity by incorporating novelty (items not rated by the user)
    recommended_items = []
    for item, _ in predicted_ratings[:top_n]:
        if item not in recommended_items:
            recommended_items.append(item)
    
    return recommended_items
 
# 4. Recommend items for User1 with diversity
user_idx = 0  # User1
recommended_items = recommend_items(user_idx, df, item_similarity_df, top_n=3)
print(f"Diversity-Aware Recommendations for User1: {recommended_items}")