"""
Project 321. Collaborative filtering implementation
Description:
Collaborative filtering is one of the most common methods for building recommendation systems. It predicts the interests of a user by collecting preferences or taste information from many users. There are two main types:

User-based: Recommends items by finding similar users

Item-based: Recommends items similar to the ones the user has liked

We’ll implement a user-based collaborative filtering method using cosine similarity between users.

About:
✅ What It Does:
Simulates a user-item rating matrix with random values

Computes cosine similarity between users

Recommends items for User1 based on ratings from similar users
"""

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
 
# 1. Simulate user-item ratings (random data)
np.random.seed(42)
users = ['User1', 'User2', 'User3', 'User4', 'User5']
items = ['Item1', 'Item2', 'Item3', 'Item4', 'Item5']
ratings = np.random.randint(1, 6, size=(5, 5))  # Random ratings between 1 and 5
 
# Create DataFrame
df = pd.DataFrame(ratings, index=users, columns=items)
print("Ratings Matrix:\n", df)
 
# 2. Calculate cosine similarity between users
user_similarity = cosine_similarity(df)
user_similarity_df = pd.DataFrame(user_similarity, index=users, columns=users)
 
# 3. Recommend items for User1 based on similarity
user1_ratings = df.loc['User1']
similar_users = user_similarity_df['User1'].sort_values(ascending=False).index[1:]  # Exclude User1 itself
 
# Get weighted ratings from similar users
recommendations = []
for item in df.columns:
    weighted_sum = 0
    total_similarity = 0
    for user in similar_users:
        if pd.notna(df.loc[user, item]):
            weighted_sum += user_similarity_df.loc['User1', user] * df.loc[user, item]
            total_similarity += user_similarity_df.loc['User1', user]
    if total_similarity > 0:
        recommendations.append((item, weighted_sum / total_similarity))
    else:
        recommendations.append((item, 0))
 
recommendations = sorted(recommendations, key=lambda x: x[1], reverse=True)
print("\nRecommendations for User1:")
for item, score in recommendations:
    print(f"{item}: {score:.2f}")