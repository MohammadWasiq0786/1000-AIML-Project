"""
Project 345. Alternating least squares implementation
Description:
Alternating Least Squares (ALS) is a popular matrix factorization technique used in collaborative filtering recommendation systems. It works by factorizing the user-item interaction matrix into two lower-dimensional matrices (user and item matrices), which can then be used to predict missing entries (e.g., user-item interactions). ALS is particularly effective when working with implicit feedback data.

In this project, we’ll implement ALS to perform matrix factorization for recommendations based on implicit user-item interactions.

About:
✅ What It Does:
Simulates implicit feedback data for user-item interactions

Implements ALS for matrix factorization, updating user and item matrices iteratively to minimize the error between predicted and actual ratings

Recommends items based on predicted ratings for a given user
"""

import numpy as np
import pandas as pd
 
# 1. Simulate user-item ratings matrix (implicit feedback: 1 for liked items)
users = ['User1', 'User2', 'User3', 'User4', 'User5']
items = ['Item1', 'Item2', 'Item3', 'Item4', 'Item5']
ratings = np.array([
    [1, 1, 0, 0, 1],
    [1, 0, 0, 1, 0],
    [0, 1, 0, 1, 1],
    [0, 0, 1, 1, 1],
    [1, 1, 0, 0, 0]
])
 
df = pd.DataFrame(ratings, index=users, columns=items)
 
# 2. Implementing ALS for matrix factorization
class ALS:
    def __init__(self, n_users, n_items, embedding_dim=4, reg_param=0.1, n_iter=10):
        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = embedding_dim
        self.reg_param = reg_param
        self.n_iter = n_iter
        self.user_matrix = np.random.rand(n_users, embedding_dim)
        self.item_matrix = np.random.rand(n_items, embedding_dim)
 
    def fit(self, df):
        for epoch in range(self.n_iter):
            # Update user matrix (fix item matrix)
            for user_idx in range(self.n_users):
                rated_items = np.where(df.iloc[user_idx] > 0)[0]
                A = self.item_matrix[rated_items]
                Y = df.iloc[user_idx, rated_items]
                X = np.linalg.inv(np.dot(A.T, A) + self.reg_param * np.eye(self.embedding_dim)).dot(A.T).dot(Y)
                self.user_matrix[user_idx] = X
 
            # Update item matrix (fix user matrix)
            for item_idx in range(self.n_items):
                rated_users = np.where(df.iloc[:, item_idx] > 0)[0]
                B = self.user_matrix[rated_users]
                Y = df.iloc[rated_users, item_idx]
                X = np.linalg.inv(np.dot(B.T, B) + self.reg_param * np.eye(self.embedding_dim)).dot(B.T).dot(Y)
                self.item_matrix[item_idx] = X
 
            # Calculate and print the loss for monitoring
            loss = 0
            for user_idx in range(self.n_users):
                for item_idx in range(self.n_items):
                    if df.iloc[user_idx, item_idx] > 0:  # Only consider rated items
                        prediction = np.dot(self.user_matrix[user_idx], self.item_matrix[item_idx].T)
                        loss += (df.iloc[user_idx, item_idx] - prediction) ** 2
            print(f"Epoch {epoch+1}/{self.n_iter}, Loss: {loss:.4f}")
 
    def predict(self, user_idx, item_idx):
        return np.dot(self.user_matrix[user_idx], self.item_matrix[item_idx].T)
 
    def recommend(self, user_idx, top_n=3):
        predicted_ratings = []
        for item_idx in range(self.n_items):
            predicted_rating = self.predict(user_idx, item_idx)
            predicted_ratings.append((item_idx, predicted_rating))
        
        predicted_ratings.sort(key=lambda x: x[1], reverse=True)
        return [items[idx] for idx, _ in predicted_ratings[:top_n]]
 
# 3. Train the ALS model
als = ALS(n_users=len(users), n_items=len(items), embedding_dim=4, n_iter=10)
als.fit(df)
 
# 4. Recommend top 3 items for User1
user_idx = 0  # User1
recommended_items = als.recommend(user_idx)
print(f"ALS Recommendations for User1: {recommended_items}")