"""
Project 341. Wide & deep learning for recommendations
Description:
Wide & Deep Learning combines the benefits of wide models (which capture memorization of data) and deep models (which capture generalization). This approach is particularly useful for recommendation systems where:

Wide models memorize specific user-item interactions

Deep models generalize by learning complex patterns and interactions

In this project, we’ll build a Wide & Deep learning model that combines both types of models to make better recommendations.

About:
✅ What It Does:
Combines a wide model (linear user-item embeddings) and a deep model (neural network with embeddings) into a Wide & Deep architecture

Uses embedding layers for both users and items to capture latent features

Recommends items by combining the outputs of both models (wide and deep) for more robust predictions
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
 
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
 
# 2. Define the Wide & Deep model
class WideAndDeepModel(nn.Module):
    def __init__(self, n_users, n_items, embedding_dim=3):
        super(WideAndDeepModel, self).__init__()
        
        # Wide model: A simple linear model
        self.linear_user = nn.Embedding(n_users, 1)
        self.linear_item = nn.Embedding(n_items, 1)
        
        # Deep model: A neural network
        self.deep_user = nn.Embedding(n_users, embedding_dim)
        self.deep_item = nn.Embedding(n_items, embedding_dim)
        self.deep_fc = nn.Sequential(
            nn.Linear(embedding_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def forward(self, user_idx, item_idx):
        # Wide model: Linear combination of user and item embeddings
        wide_out = self.linear_user(user_idx) + self.linear_item(item_idx)
        
        # Deep model: Learn interactions via embeddings and feedforward network
        deep_user_emb = self.deep_user(user_idx)
        deep_item_emb = self.deep_item(item_idx)
        deep_out = self.deep_fc(torch.cat([deep_user_emb, deep_item_emb], dim=-1))
        
        # Combine wide and deep model outputs
        return wide_out + deep_out
 
# 3. Prepare the data
class RecommendationDataset(torch.utils.data.Dataset):
    def __init__(self, df):
        self.data = []
        self.n_users = len(df)
        self.n_items = len(df.columns)
        for user_idx in range(self.n_users):
            for item_idx in range(self.n_items):
                if df.iloc[user_idx, item_idx] > 0:
                    self.data.append((user_idx, item_idx, df.iloc[user_idx, item_idx]))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        user_idx, item_idx, rating = self.data[idx]
        return torch.tensor(user_idx), torch.tensor(item_idx), torch.tensor(rating, dtype=torch.float)
 
# 4. Train the model
dataset = RecommendationDataset(df)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True)
 
model = WideAndDeepModel(n_users=len(users), n_items=len(items), embedding_dim=3)
loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
 
# 5. Train the model for 20 epochs
num_epochs = 20
for epoch in range(num_epochs):
    epoch_loss = 0
    for user_idx, item_idx, rating in dataloader:
        optimizer.zero_grad()
        prediction = model(user_idx, item_idx)
        loss = loss_fn(prediction, rating)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss/len(dataloader):.4f}")
 
# 6. Recommend items for User1 (index 0)
user_idx = torch.tensor([0])  # User1
predictions = []
for item_idx in range(len(items)):
    item_idx_tensor = torch.tensor([item_idx])
    predicted_rating = model(user_idx, item_idx_tensor).item()
    predictions.append((items[item_idx], predicted_rating))
 
# Sort predictions by rating and display top recommendations
predictions.sort(key=lambda x: x[1], reverse=True)
print("\nRecommendations for User1:")
for item, pred in predictions[:3]:
    print(f"{item}: Predicted Rating = {pred:.2f}")