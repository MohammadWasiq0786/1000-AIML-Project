"""
Project 338. Attention mechanisms for recommendations
Description:
Attention mechanisms allow models to focus on the most relevant parts of the input data. In recommendation systems, attention can help the model:

Weigh user preferences for different items

Focus on key features of items that matter most for recommendations

Improve relevance by assigning different attention scores to features like price, genre, or rating

In this project, we’ll implement a simple attention mechanism in a recommendation system, allowing the model to focus on more important features when making suggestions.

About:
✅ What It Does:
Simulates a user-item ratings matrix and builds an embedding-based recommendation system

Implements an attention mechanism to give different importance to user-item interactions

Uses learned attention weights to focus on the most important aspects when making recommendations
"""



import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
 
# 1. Simulate user-item ratings matrix (ratings for different items by users)
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
 
# 2. Define Attention Mechanism
class Attention(nn.Module):
    def __init__(self, input_size):
        super(Attention, self).__init__()
        self.attn_weights = nn.Parameter(torch.randn(input_size))  # Learnable attention weights
 
    def forward(self, x):
        # Compute attention scores by applying weights to the input
        attn_scores = torch.matmul(x, self.attn_weights)
        attn_weights = torch.softmax(attn_scores, dim=0)  # Normalize the scores
        weighted_input = x * attn_weights  # Apply attention weights
        return weighted_input.sum(dim=0)  # Aggregate the weighted inputs
 
# 3. Define Recommendation Model with Attention
class RecommendationModel(nn.Module):
    def __init__(self, n_users, n_items, embedding_dim=5):
        super(RecommendationModel, self).__init__()
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)
        self.attention = Attention(embedding_dim)  # Attention mechanism
 
    def forward(self, user_idx, item_idx):
        user_emb = self.user_embedding(user_idx)
        item_emb = self.item_embedding(item_idx)
        combined_emb = torch.stack([user_emb, item_emb], dim=0)  # Combine user and item embeddings
        attention_output = self.attention(combined_emb)  # Apply attention mechanism
        return attention_output
 
# 4. Initialize model, loss function, and optimizer
n_users, n_items = df.shape
model = RecommendationModel(n_users, n_items)
loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
 
# 5. Train the model
num_epochs = 10
for epoch in range(num_epochs):
    epoch_loss = 0
    for user_idx in range(len(users)):
        for item_idx in range(len(items)):
            if df.iloc[user_idx, item_idx] > 0:  # Only train on rated items
                optimizer.zero_grad()
                user_tensor = torch.tensor([user_idx])
                item_tensor = torch.tensor([item_idx])
                predicted_rating = model(user_tensor, item_tensor)
                actual_rating = torch.tensor(df.iloc[user_idx, item_idx], dtype=torch.float)
                loss = loss_fn(predicted_rating, actual_rating)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
 
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss/len(users)/len(items):.4f}")
 
# 6. Make predictions for User1 on all items
user_idx = 0  # User1
predicted_ratings = []
for item_idx in range(len(items)):
    user_tensor = torch.tensor([user_idx])
    item_tensor = torch.tensor([item_idx])
    predicted_ratings.append(model(user_tensor, item_tensor).item())
 
# 7. Output predicted ratings for User1
print(f"\nPredicted Ratings for User1:")
for item, pred in zip(items, predicted_ratings):
    print(f"{item}: Predicted Rating = {pred:.2f}")