"""
Project 325. Deep learning for recommendation systems
Description:
Deep learning can capture complex, non-linear relationships in data, making it particularly powerful for recommendation systems. We’ll use a feedforward neural network to model the interaction between users and items based on their features. The model will predict ratings or preferences by learning latent representations.

In this project, we’ll implement a simple neural collaborative filtering (NCF) model using a multi-layer perceptron (MLP).

About:
✅ What It Does:
Simulates a user-item ratings matrix for a recommendation system

Builds a neural collaborative filtering model using embedding layers for users and items

Trains the model using mean squared error (MSE) loss

Predicts ratings for User1 across all items
"""


import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
 
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
 
# 2. Prepare data for deep learning model (convert to user-item pairs)
class RecommendationDataset(Dataset):
    def __init__(self, ratings_matrix):
        self.ratings = ratings_matrix
        self.user_item_pairs = []
        for user_idx in range(ratings_matrix.shape[0]):
            for item_idx in range(ratings_matrix.shape[1]):
                if ratings_matrix[user_idx, item_idx] > 0:
                    self.user_item_pairs.append((user_idx, item_idx, ratings_matrix[user_idx, item_idx]))
 
    def __len__(self):
        return len(self.user_item_pairs)
 
    def __getitem__(self, idx):
        user_idx, item_idx, rating = self.user_item_pairs[idx]
        return torch.tensor(user_idx), torch.tensor(item_idx), torch.tensor(rating, dtype=torch.float)
 
dataset = RecommendationDataset(ratings)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
 
# 3. Define the neural network model (MLP)
class NeuralCollaborativeFiltering(nn.Module):
    def __init__(self, n_users, n_items, embedding_dim=8):
        super(NeuralCollaborativeFiltering, self).__init__()
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)
        self.fc1 = nn.Linear(embedding_dim * 2, 64)
        self.fc2 = nn.Linear(64, 1)
 
    def forward(self, user_idx, item_idx):
        user_emb = self.user_embedding(user_idx)
        item_emb = self.item_embedding(item_idx)
        x = torch.cat([user_emb, item_emb], dim=-1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x.squeeze()
 
# 4. Initialize the model, loss function, and optimizer
n_users, n_items = df.shape
model = NeuralCollaborativeFiltering(n_users, n_items)
loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
 
# 5. Train the model
num_epochs = 10
for epoch in range(num_epochs):
    epoch_loss = 0
    for user_idx, item_idx, rating in dataloader:
        optimizer.zero_grad()
        pred = model(user_idx, item_idx)
        loss = loss_fn(pred, rating)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
 
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss/len(dataloader):.4f}")
 
# 6. Make predictions for a user (e.g., User1)
user_id = torch.tensor([0])  # User1
item_ids = torch.arange(n_items)  # All items
predictions = model(user_id.repeat(n_items), item_ids)
 
# Print predicted ratings for User1
print("\nPredicted Ratings for User1:")
for item_idx, pred in enumerate(predictions.detach().numpy()):
    print(f"Item{item_idx+1}: Predicted Rating = {pred:.2f}")