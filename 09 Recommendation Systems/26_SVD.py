"""
Project 346. SVD++ implementation
Description:
SVD++ is an enhancement of the traditional Singular Value Decomposition (SVD) that incorporates implicit feedback (e.g., clicks, views, or purchases) along with explicit ratings. SVD++ uses additional latent factors to model the influence of implicit feedback, improving recommendation quality, especially when explicit ratings are sparse.

In this project, we’ll implement SVD++ for matrix factorization and use it to make recommendations.

About:
✅ What It Does:
Implements SVD++ by incorporating both explicit and implicit feedback for better recommendations

Utilizes item and user embeddings to capture latent factors, and bias terms for each user and item

Trains the model to minimize the error between predicted and actual ratings using gradient descent

Recommends items based on learned item-user interactions and implicit feedback from other items the user has interacted with
"""


import numpy as np
import pandas as pd
 
# 1. Simulate user-item ratings matrix (implicit feedback: 1 for liked items)
users = ['User1', 'User2', 'User3', 'User4', 'User5']
items = ['Item1', 'Item2', 'Item3', 'Item4', 'Item5']
ratings = np.array([
    [5, 4, 0, 0, 3],
    [4, 0, 0, 3, 2],
    [1, 1, 0, 5, 4],
    [0, 0, 5, 4, 4],
    [2, 3, 0, 1, 0]
])
 
df = pd.DataFrame(ratings, index=users, columns=items)
 
# 2. Define the SVD++ model
class SVDPlusPlus:
    def __init__(self, n_users, n_items, embedding_dim=4, reg_param=0.1, n_iter=10, alpha=0.02):
        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = embedding_dim
        self.reg_param = reg_param
        self.n_iter = n_iter
        self.alpha = alpha
 
        # Initialize user and item embeddings
        self.user_embedding = np.random.rand(n_users, embedding_dim)
        self.item_embedding = np.random.rand(n_items, embedding_dim)
        self.item_bias = np.random.rand(n_items)  # Bias terms for items
        self.user_bias = np.random.rand(n_users)  # Bias terms for users
 
        # Implicit feedback terms (SVD++)
        self.y = np.random.rand(n_items, embedding_dim)  # Implicit feedback embeddings
        self.i_count = np.zeros(n_users)  # Implicit feedback count for each user
 
    def fit(self, df):
        for epoch in range(self.n_iter):
            loss = 0
            for user_idx in range(self.n_users):
                for item_idx in range(self.n_items):
                    if df.iloc[user_idx, item_idx] > 0:  # Only consider rated items
                        # Calculate the implicit feedback count (number of interactions with items)
                        implicit_feedback = np.where(df.iloc[user_idx] > 0)[0]
                        self.i_count[user_idx] = len(implicit_feedback)
                        
                        # Calculate the prediction for this user-item pair
                        x_ui = np.dot(self.user_embedding[user_idx], self.item_embedding[item_idx]) + self.user_bias[user_idx] + self.item_bias[item_idx]
                        
                        # Add implicit feedback to the item factorization
                        implicit_sum = np.sum(self.y[implicit_feedback], axis=0)
                        x_ui += np.dot(self.user_embedding[user_idx], implicit_sum)
                        
                        # Calculate error
                        error = df.iloc[user_idx, item_idx] - x_ui
                        loss += error**2
 
                        # Update the model parameters using gradient descent
                        self.user_embedding[user_idx] += self.alpha * (error * self.item_embedding[item_idx] - self.reg_param * self.user_embedding[user_idx])
                        self.item_embedding[item_idx] += self.alpha * (error * self.user_embedding[user_idx] - self.reg_param * self.item_embedding[item_idx])
                        self.user_bias[user_idx] += self.alpha * (error - self.reg_param * self.user_bias[user_idx])
                        self.item_bias[item_idx] += self.alpha * (error - self.reg_param * self.item_bias[item_idx])
                        
                        for i in implicit_feedback:
                            self.y[i] += self.alpha * (error * self.user_embedding[user_idx] - self.reg_param * self.y[i])
 
            print(f"Epoch {epoch+1}/{self.n_iter}, Loss: {loss:.4f}")
 
    def predict(self, user_idx, item_idx):
        implicit_feedback = np.where(df.iloc[user_idx] > 0)[0]
        implicit_sum = np.sum(self.y[implicit_feedback], axis=0)
        prediction = np.dot(self.user_embedding[user_idx], self.item_embedding[item_idx]) + self.user_bias[user_idx] + self.item_bias[item_idx]
        prediction += np.dot(self.user_embedding[user_idx], implicit_sum)
        return prediction
 
    def recommend(self, user_idx, top_n=3):
        predictions = []
        for item_idx in range(self.n_items):
            predicted_rating = self.predict(user_idx, item_idx)
            predictions.append((item_idx, predicted_rating))
 
        predictions.sort(key=lambda x: x[1], reverse=True)
        return [items[idx] for idx, _ in predictions[:top_n]]
 
# 3. Train the SVD++ model
svdpp = SVDPlusPlus(n_users=len(users), n_items=len(items), embedding_dim=4, n_iter=10, alpha=0.02)
svdpp.fit(df)
 
# 4. Recommend top 3 items for User1
user_idx = 0  # User1
recommended_items = svdpp.recommend(user_idx)
print(f"SVD++ Recommendations for User1: {recommended_items}")