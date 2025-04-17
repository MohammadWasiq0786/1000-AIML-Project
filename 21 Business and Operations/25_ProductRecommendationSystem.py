"""
Project 825. Product Recommendation System

A product recommendation system suggests items to users based on their past behavior or similarities with other users. In this basic project, we’ll build a user-based collaborative filtering system using a user-item ratings matrix and cosine similarity to recommend products.

This collaborative filtering system uses user similarity to infer preferences and suggest new products. It can be scaled with matrix factorization or neural recommenders for better performance on large datasets.
"""

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
 
# Simulated user-item rating matrix
# Rows = Users, Columns = Products, Values = Ratings (1–5), NaN = not rated
data = {
    'Product_A': [5, 4, np.nan, 2, np.nan],
    'Product_B': [3, np.nan, 2, 5, 4],
    'Product_C': [np.nan, 2, 4, np.nan, 3],
    'Product_D': [1, 2, 3, 5, 4]
}
user_ids = ['User1', 'User2', 'User3', 'User4', 'User5']
df = pd.DataFrame(data, index=user_ids)
 
# Fill NaNs with 0s (optional: use mean/ALS for better imputation)
filled = df.fillna(0)
 
# Compute cosine similarity between users
similarity = cosine_similarity(filled)
similarity_df = pd.DataFrame(similarity, index=user_ids, columns=user_ids)
 
# Function to recommend products for a given user
def recommend_products(user_id, top_n=2):
    user_index = df.index.get_loc(user_id)
    similar_users = similarity[user_index]
    
    # Weighted ratings based on similarity
    weighted_ratings = np.dot(similar_users, filled.values)
    user_seen = ~df.loc[user_id].isna()
    weighted_ratings[user_seen.values] = 0  # Exclude already rated items
    
    # Recommend top N unrated products
    product_indices = np.argsort(weighted_ratings)[::-1][:top_n]
    recommended = df.columns[product_indices]
    return list(recommended)
 
# Example: recommend for User3
recommendations = recommend_products('User3', top_n=2)
print(f"Recommended products for User3: {recommendations}")