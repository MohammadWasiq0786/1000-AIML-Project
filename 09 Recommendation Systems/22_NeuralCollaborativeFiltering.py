"""
Project 342. Neural collaborative filtering
Description:
Neural Collaborative Filtering (NCF) is an advanced recommendation technique that uses deep learning to model the interaction between users and items. Traditional collaborative filtering relies on linear models, but NCF introduces neural networks to capture complex, non-linear relationships between users and items, improving recommendation performance.

In this project, we’ll build an NCF model using multi-layer perceptrons (MLPs) to predict ratings and generate personalized recommendations.

About:
✅ What It Does:
Simulates a user-item ratings matrix for training the model

Defines a Neural Collaborative Filtering (NCF) model using MLP layers to capture non-linear user-item interactions

Trains the model using mean squared error (MSE) loss and Adam optimizer

Recommends items based on learned interactions for User1
"""


import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
 
# 1. Simulate user-item ratings matrix
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
 
# 2. Define Neural Collaborative Filtering Model (MLP-based)
class NeuralCollaborativeFiltering(nn.Module):
    def __init__(self, n_users, n_items, embedding_dim=4):
        super(NeuralCollaborativeFiltering, self).__init__()
        
        # Embeddings for users and items
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)
        
        # MLP layers for the interaction between user and item embeddings
        self.fc1 = nn.Linear(embedding_dim * 2, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
    
    def forward(self, user_idx, item_idx):
        user_emb = self.user_embedding(user_idx)
        item_emb = self.item_embedding(item_idx)
        x = torch.cat([user_emb, item_emb], dim=-1)  # Concatenate user and item embeddings
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x.squeeze()  # Remove extra dimension
 
# 3. Prepare DataLoader for training
class RecommendationDataset(torch.utils.data.Dataset):
    def __init__(self, df):
        self.data = []
        self.n_users = len(df)
        self.n_items = len(df.columns)
        for user_idx in range(self.n_users):
            for item_idx in range(self.n_items):
                if df.iloc[user_idx, item_idx] > 0:  # Only train on rated items
                    self.data.append((user_idx, item_idx, df.iloc[user_idx, item_idx]))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        user_idx, item_idx, rating = self.data[idx]
        return torch.tensor(user_idx), torch.tensor(item_idx), torch.tensor(rating, dtype=torch.float)
 
# 4. Initialize the model, loss function, and optimizer
dataset = RecommendationDataset(df)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True)
 
model = NeuralCollaborativeFiltering(n_users=len(users), n_items=len(items), embedding_dim=4)
loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
 
# 5. Train the model for 10 epochs
num_epochs = 10
for epoch in range(num_epochs):
    epoch_loss = 0
    for user_idx, item_idx, rating in dataloader:
        optimizer.zero_grad()
        predicted_rating = model(user_idx, item_idx)
        loss = loss_fn(predicted_rating, rating)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss/len(dataloader):.4f}")
 
# 6. Make predictions for User1 (index 0) for all items
user_idx = torch.tensor([0])  # User1
predictions = []
for item_idx in range(len(items)):
    item_idx_tensor = torch.tensor([item_idx])
    predicted_rating = model(user_idx, item_idx_tensor).item()
    predictions.append((items[item_idx], predicted_rating))
 
# 7. Sort predictions by rating and display top recommendations
predictions.sort(key=lambda x: x[1], reverse=True)
print("\nNeural Collaborative Filtering Recommendations for User1:")
for item, pred in predictions[:3]:
    print(f"{item}: Predicted Rating = {pred:.2f}")