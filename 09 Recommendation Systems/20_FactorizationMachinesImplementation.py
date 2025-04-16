"""
Project 340. Factorization machines implementation
Description:
Factorization machines (FM) are a generalization of matrix factorization techniques and are particularly effective in capturing pairwise interactions between features. They work well for:

High-dimensional sparse datasets

Recommendation systems where interactions between users and items need to be modeled

In this project, we’ll implement a factorization machine model to capture interactions between users and items and make recommendations based on those interactions.

About:
✅ What It Does:
Simulates a user-item ratings matrix and prepares the data for factorization machine learning

Builds a factorization machine model using user-item embeddings and stochastic gradient descent (SGD)

Trains the model to learn latent factors (hidden features) and interactions between users and items

Recommends top items based on the learned embeddings for User1
"""


import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
 
# 1. Simulate user-item ratings matrix (user-item interactions)
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
 
# 2. Prepare the data for Factorization Machine (FM)
# Convert the ratings matrix into a feature matrix with interactions
user_item_data = []
 
for user_idx, user in enumerate(users):
    for item_idx, item in enumerate(items):
        if df.iloc[user_idx, item_idx] > 0:  # If the user rated the item
            user_item_data.append([user_idx, item_idx, df.iloc[user_idx, item_idx]])
 
# Convert to a DataFrame for easier handling
user_item_df = pd.DataFrame(user_item_data, columns=['user_idx', 'item_idx', 'rating'])
 
# 3. Factorization Machines model using SGD (Stochastic Gradient Descent)
# Factorization machines model using interactions between user-item pairs
class FactorizationMachine:
    def __init__(self, n_users, n_items, latent_dim=3):
        self.n_users = n_users
        self.n_items = n_items
        self.latent_dim = latent_dim
        self.user_embeddings = np.random.randn(n_users, latent_dim)
        self.item_embeddings = np.random.randn(n_items, latent_dim)
    
    def predict(self, user_idx, item_idx):
        # Compute the predicted rating based on the interaction of user-item embedding
        return np.dot(self.user_embeddings[user_idx], self.item_embeddings[item_idx])
    
    def fit(self, user_item_df, n_epochs=100, learning_rate=0.01, reg_param=0.1):
        for epoch in range(n_epochs):
            total_loss = 0
            for _, row in user_item_df.iterrows():
                user_idx = row['user_idx']
                item_idx = row['item_idx']
                actual_rating = row['rating']
                
                predicted_rating = self.predict(user_idx, item_idx)
                error = actual_rating - predicted_rating
                
                # Gradient descent update for the embeddings
                self.user_embeddings[user_idx] += learning_rate * (error * self.item_embeddings[item_idx] - reg_param * self.user_embeddings[user_idx])
                self.item_embeddings[item_idx] += learning_rate * (error * self.user_embeddings[user_idx] - reg_param * self.item_embeddings[item_idx])
                
                total_loss += error ** 2
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {total_loss}")
                
    def recommend(self, user_idx, top_n=3):
        # Recommend top N items for a user based on learned embeddings
        predictions = []
        for item_idx in range(self.n_items):
            predicted_rating = self.predict(user_idx, item_idx)
            predictions.append((item_idx, predicted_rating))
        
        predictions.sort(key=lambda x: x[1], reverse=True)
        recommended_items = [items[idx] for idx, _ in predictions[:top_n]]
        return recommended_items
 
# 4. Initialize and train the Factorization Machine model
fm = FactorizationMachine(n_users=len(users), n_items=len(items), latent_dim=3)
fm.fit(user_item_df, n_epochs=100, learning_rate=0.01, reg_param=0.1)
 
# 5. Recommend items for User1 based on learned embeddings
recommended_items = fm.recommend(user_idx=0, top_n=3)  # User1 (index 0)
print("\nFactorization Machine Recommendations for User1:", recommended_items)