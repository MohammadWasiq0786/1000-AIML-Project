"""
Project 324. Matrix factorization for recommendations
Description:
Matrix Factorization is a powerful technique used in recommendation systems, especially in collaborative filtering. It decomposes the user-item interaction matrix into lower-rank matrices to uncover latent factors (features) that explain observed ratings.

We’ll implement Singular Value Decomposition (SVD), a common matrix factorization technique, to generate recommendations based on latent factors.

About:
✅ What It Does:
Simulates a user-item ratings matrix

Performs Singular Value Decomposition (SVD) for matrix factorization

Reconstructs the ratings matrix to generate latent factors

Recommends unrated items for User1 based on the predicted ratings
"""


import numpy as np
import pandas as pd
from scipy.sparse.linalg import svds
 
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
print("User-Item Ratings Matrix:\n", df)
 
# 2. Perform SVD (Matrix Factorization)
U, sigma, Vt = svds(ratings, k=2)  # k is the number of latent factors
 
# 3. Reconstruct the ratings matrix (approximated)
sigma = np.diag(sigma)
approx_ratings = np.dot(np.dot(U, sigma), Vt)
 
# 4. Convert the approximated matrix back to a DataFrame
approx_df = pd.DataFrame(approx_ratings, columns=items, index=users)
print("\nApproximated Ratings Matrix using SVD:\n", approx_df)
 
# 5. Recommend items for a given user (e.g., User1)
user_idx = 0  # User1
user_ratings = approx_df.iloc[user_idx].values
unrated_items = np.where(df.iloc[user_idx].values == 0)[0]  # Find unrated items
 
# Predict ratings for unrated items
predicted_ratings = [(items[i], user_ratings[i]) for i in unrated_items]
 
# 6. Sort and recommend top items
recommended_items = sorted(predicted_ratings, key=lambda x: x[1], reverse=True)
print("\nRecommendations for User1 based on Matrix Factorization:")
for item, rating in recommended_items:
    print(f"{item}: Predicted Rating = {rating:.2f}")