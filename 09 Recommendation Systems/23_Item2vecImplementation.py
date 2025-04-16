"""
Project 343. Item2vec implementation
Description:
Item2Vec is a model inspired by Word2Vec (a popular word embedding technique), where items are embedded in a vector space based on their co-occurrence patterns. In recommendation systems, the goal is to represent items as dense vectors, where similar items are mapped closer together in the vector space. This allows us to find item similarities and generate recommendations.

In this project, we’ll implement Item2Vec to create item embeddings and use them for item-based recommendations.

About:
✅ What It Does:
Simulates a user-item ratings matrix and prepares user-item pairs for training

Uses Item2Vec to learn item embeddings based on user-item interactions and negative sampling

Recommends items based on cosine similarity between item embeddings, which captures the semantic similarity between items
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder
 
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
 
# 2. Prepare data for Item2Vec (convert user-item interactions into pairs)
user_item_pairs = []
 
for user_idx, user in enumerate(users):
    for item_idx, item in enumerate(items):
        if df.iloc[user_idx, item_idx] > 0:  # Only consider rated items
            user_item_pairs.append((user, items[item_idx]))
 
# 3. Encode items and users using LabelEncoder
item_encoder = LabelEncoder()
item_encoder.fit(items)
encoded_items = item_encoder.transform(items)
 
# 4. Define the Item2Vec model (skip-gram model for item embeddings)
class Item2Vec(nn.Module):
    def __init__(self, n_items, embedding_dim=4):
        super(Item2Vec, self).__init__()
        self.item_embedding = nn.Embedding(n_items, embedding_dim)
        self.context_embedding = nn.Embedding(n_items, embedding_dim)
 
    def forward(self, item_idx, context_idx):
        item_emb = self.item_embedding(item_idx)
        context_emb = self.context_embedding(context_idx)
        score = torch.sum(item_emb * context_emb, dim=-1)  # Dot product between item and context embeddings
        return score
 
# 5. Training the Item2Vec model using negative sampling (simplified)
model = Item2Vec(n_items=len(items), embedding_dim=4)
loss_fn = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
 
# 6. Create negative samples (random items as negative context)
def create_negative_samples(user_item_pairs, n_negative=1):
    negative_samples = []
    for pair in user_item_pairs:
        user, item = pair
        negative_items = np.random.choice(items, size=n_negative, replace=False)
        for neg_item in negative_items:
            negative_samples.append((item, neg_item))  # (positive item, negative item)
    return negative_samples
 
# 7. Train the model
num_epochs = 100
for epoch in range(num_epochs):
    epoch_loss = 0
    negative_samples = create_negative_samples(user_item_pairs)
 
    for item, context_item in negative_samples:
        item_idx = torch.tensor([item_encoder.transform([item])[0]])
        context_idx = torch.tensor([item_encoder.transform([context_item])[0]])
        
        optimizer.zero_grad()
        score = model(item_idx, context_idx)
        
        # Positive pair score (we use sigmoid to interpret as probability)
        label = torch.tensor([1.0])  # Positive sample (real interaction)
        
        loss = loss_fn(score, label)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
 
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss/len(negative_samples):.4f}")
 
# 8. Generate item recommendations (items with the most similar embeddings)
item_idx = torch.tensor([item_encoder.transform(['Item1'])[0]])
item_embedding = model.item_embedding(item_idx).detach().numpy()
 
# Calculate cosine similarity between Item1 and all other items
item_similarities = []
for i in range(len(items)):
    item_sim = torch.nn.functional.cosine_similarity(
        model.item_embedding(item_idx), model.item_embedding(torch.tensor([i])))
    item_similarities.append((items[i], item_sim.item()))
 
# Sort items based on similarity to Item1
item_similarities.sort(key=lambda x: x[1], reverse=True)
 
# 9. Print recommendations for Item1
print("\nRecommended items for Item1 based on Item2Vec:")
for item, sim in item_similarities[:3]:
    print(f"{item}: Similarity = {sim:.2f}")