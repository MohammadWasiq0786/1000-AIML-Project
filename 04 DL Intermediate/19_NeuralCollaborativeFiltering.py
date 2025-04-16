"""
Project 139. Neural collaborative filtering
Description:
Neural Collaborative Filtering (NCF) combines the strengths of matrix factorization and deep neural networks to model user-item interactions. It learns non-linear relationships between users and items through embeddings and multi-layer perceptrons, leading to more accurate and expressive recommendation systems.

About:
🧠 What This Project Demonstrates:
Implements Neural Collaborative Filtering with user/item embeddings

Trains with implicit feedback data (e.g., clicks, likes)

Models non-linear user-item interactions via a neural MLP
"""

# Install if not already: pip install torch pandas numpy scikit-learn
 
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
 
# Sample data: user_id, item_id, rating
data = {
    "user_id": [0, 0, 1, 1, 2, 2, 3, 4],
    "item_id": [1, 2, 2, 3, 1, 3, 0, 1],
    "rating":  [1, 1, 1, 1, 1, 1, 1, 1]  # Implicit feedback (1 = interaction)
}
df = pd.DataFrame(data)
 
# Get unique counts
num_users = df['user_id'].nunique()
num_items = df['item_id'].nunique()
 
# Train/test split
train_data, test_data = train_test_split(df, test_size=0.25)
 
# Neural Collaborative Filtering Model
class NCF(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=8):
        super(NCF, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
 
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
 
    def forward(self, user_ids, item_ids):
        u_embed = self.user_embedding(user_ids)
        i_embed = self.item_embedding(item_ids)
        x = torch.cat([u_embed, i_embed], dim=1)
        return self.mlp(x)
 
# Instantiate model, loss, optimizer
model = NCF(num_users, num_items).float()
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)
 
# Convert training data to tensors
user_tensor = torch.tensor(train_data["user_id"].values, dtype=torch.long)
item_tensor = torch.tensor(train_data["item_id"].values, dtype=torch.long)
label_tensor = torch.tensor(train_data["rating"].values, dtype=torch.float32)
 
# Train (one epoch for demo)
model.train()
for epoch in range(1):
    preds = model(user_tensor, item_tensor).squeeze()
    loss = criterion(preds, label_tensor)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(f"✅ Epoch {epoch+1} - Loss: {loss.item():.4f}")
 
# Test prediction
model.eval()
with torch.no_grad():
    test_user = torch.tensor(test_data["user_id"].values, dtype=torch.long)
    test_item = torch.tensor(test_data["item_id"].values, dtype=torch.long)
    test_preds = model(test_user, test_item).squeeze()
    print("\n📢 Test Predictions (interaction scores):", test_preds.tolist())