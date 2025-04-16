"""
Project 333. Explainable recommendation system
Description:
Explainability in recommendation systems allows users to understand why they are receiving a specific recommendation. This is important for:

Trust and transparency with users

Debugging and improving the system

Regulatory compliance in some industries

In this project, we’ll implement an explainable recommendation system that provides reasons for recommendations based on item similarities and user preferences.

About:
✅ What It Does:
Calculates item-item similarity using cosine similarity

Predicts ratings for unrated items using weighted sums of similar items

Provides an explanation for each recommendation by listing top similar items contributing to the suggestion
"""

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
 
# 1. Simulate user-item ratings data
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
 
# 2. Compute item-item similarity (cosine similarity)
item_similarity = cosine_similarity(df.T)
item_similarity_df = pd.DataFrame(item_similarity, index=items, columns=items)
 
# 3. Define function to recommend items for a user with explanations
def explainable_recommendations(user_idx, df, item_similarity_df, top_n=3):
    user_ratings = df.iloc[user_idx]
    unrated_items = user_ratings[user_ratings == 0].index
    
    recommendations = []
    for item in unrated_items:
        # Calculate predicted rating (weighted sum based on item similarity)
        similar_items = item_similarity_df[item]
        weighted_sum = np.dot(user_ratings, similar_items)
        total_similarity = np.sum(similar_items)
        predicted_rating = weighted_sum / total_similarity if total_similarity > 0 else 0
        recommendations.append((item, predicted_rating, similar_items))
 
    # Sort by predicted rating
    recommendations = sorted(recommendations, key=lambda x: x[1], reverse=True)
    
    # Prepare explanation for the top recommendations
    explanations = []
    for item, rating, similarities in recommendations[:top_n]:
        # Find similar items contributing to the recommendation
        top_similar_items = sorted(list(similarities.items()), key=lambda x: x[1], reverse=True)[:3]
        explanation = f"Recommended {item} because it's similar to: " + ", ".join([f"{i[0]}" for i in top_similar_items])
        explanations.append((item, rating, explanation))
    
    return explanations
 
# 4. Recommend items for User1 with explanations
user_idx = 0  # User1
explanations = explainable_recommendations(user_idx, df, item_similarity_df, top_n=2)
print(f"Explainable Recommendations for User1:")
for item, rating, explanation in explanations:
    print(f"{item}: Predicted Rating = {rating:.2f} | Reason: {explanation}")