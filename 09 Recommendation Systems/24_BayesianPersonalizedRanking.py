"""
Project 344. Bayesian personalized ranking
Description:
Bayesian Personalized Ranking (BPR) is a pairwise ranking method used in recommendation systems. It models the preferences of users by comparing items that have been interacted with (positive samples) and those that have not (negative samples). BPR is particularly effective when:

The system focuses on ranking rather than predicting exact ratings

There is a need to model implicit feedback (like clicks, views, or purchases)

In this project, we’ll implement BPR using stochastic gradient descent (SGD) for learning item embeddings.

About:
✅ What It Does:
Simulates implicit feedback for user-item interactions

Implements Bayesian Personalized Ranking (BPR) for learning item embeddings using pairwise ranking

Trains the model using stochastic gradient descent (SGD) and binary cross-entropy loss to predict item preferences

Recommends items based on the learned rankings for User1
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
 
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
 
# 2. Define the BPR model (pairwise ranking)
class BPR(nn.Module):
    def __init__(self, n_users, n_items, embedding_dim=4):
        super(BPR, self).__init__()
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)
 
    def forward(self, user_idx, pos_item_idx, neg_item_idx):
        # Positive and negative item embeddings
        user_emb = self.user_embedding(user_idx)
        pos_item_emb = self.item_embedding(pos_item_idx)
        neg_item_emb = self.item_embedding(neg_item_idx)
 
        # Compute dot products (user-item interaction)
        pos_score = torch.sum(user_emb * pos_item_emb, dim=-1)
        neg_score = torch.sum(user_emb * neg_item_emb, dim=-1)
        
        return pos_score, neg_score
 
# 3. Prepare DataLoader for training with implicit feedback (pairwise)
class BPRDataset(torch.utils.data.Dataset):
    def __init__(self, df):
        self.data = []
        self.n_users = len(df)
        self.n_items = len(df.columns)
        for user_idx in range(self.n_users):
            for item_idx in range(self.n_items):
                if df.iloc[user_idx, item_idx] > 0:  # Positive item interactions
                    # Negative sample: randomly choose an item the user hasn't interacted with
                    neg_item_idx = np.random.choice([i for i in range(self.n_items) if df.iloc[user_idx, i] == 0])
                    self.data.append((user_idx, item_idx, neg_item_idx))
 
    def __len__(self):
        return len(self.data)
 
    def __getitem__(self, idx):
        user_idx, pos_item_idx, neg_item_idx = self.data[idx]
        return torch.tensor(user_idx), torch.tensor(pos_item_idx), torch.tensor(neg_item_idx)
 
# 4. Initialize the BPR model, loss function, and optimizer
dataset = BPRDataset(df)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True)
 
model = BPR(n_users=len(users), n_items=len(items), embedding_dim=4)
loss_fn = nn.BCEWithLogitsLoss()  # Binary cross-entropy loss
optimizer = optim.Adam(model.parameters(), lr=0.001)
 
# 5. Train the BPR model
num_epochs = 10
for epoch in range(num_epochs):
    epoch_loss = 0
    for user_idx, pos_item_idx, neg_item_idx in dataloader:
        optimizer.zero_grad()
        pos_score, neg_score = model(user_idx, pos_item_idx, neg_item_idx)
        
        # Compute the loss (maximize the difference between positive and negative scores)
        loss = loss_fn(pos_score - neg_score, torch.ones_like(pos_score))
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss/len(dataloader):.4f}")
 
# 6. Recommend items for User1 based on learned embeddings
user_idx = torch.tensor([0])  # User1
predictions = []
for item_idx in range(len(items)):
    pos_item_idx = torch.tensor([item_idx])
    neg_item_idx = torch.tensor([np.random.choice([i for i in range(len(items)) if i != item_idx])])  # Random negative sample
    pos_score, neg_score = model(user_idx, pos_item_idx, neg_item_idx)
    predictions.append((items[item_idx], pos_score.item() - neg_score.item()))  # Score = pos_score - neg_score
 
# Sort items by the learned score and display top recommendations
predictions.sort(key=lambda x: x[1], reverse=True)
print("\nRecommended items for User1 (BPR-based):")
for item, score in predictions[:3]:
    print(f"{item}: Score = {score:.2f}")