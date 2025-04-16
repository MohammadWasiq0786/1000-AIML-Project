"""
Project 309. Few-shot learning for time series
Description:
Few-shot learning enables models to learn new tasks with very limited data, often using techniques like:

Meta-learning (learning to learn)

Prototype networks

Embedding + similarity comparison

In this project, we’ll simulate few-shot time series classification using a Prototypical Network idea — computing embeddings for a few examples and classifying based on distance in embedding space.

About:
✅ What It Does:
Simulates a few-shot classification task with 5 examples per class

Learns embeddings with a simple ConvNet

Classifies queries by comparing to class prototypes in feature space

Effective when labeled data is scarce, and new classes emerge frequently
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.metrics.pairwise import euclidean_distances
 
# 1. Generate two classes of time series
def generate_class(n, length, wave_func, noise=0.1):
    return np.array([wave_func(np.linspace(0, 3 * np.pi, length)) + noise * np.random.randn(length) for _ in range(n)])
 
length = 50
class_A = generate_class(5, length, np.sin)  # Few examples (support set)
class_B = generate_class(5, length, np.cos)
 
# Queries
query_A = generate_class(10, length, np.sin)
query_B = generate_class(10, length, np.cos)
queries = np.concatenate([query_A, query_B])
query_labels = np.array([0]*10 + [1]*10)
 
# Combine support set
support = np.concatenate([class_A, class_B])
support_labels = np.array([0]*5 + [1]*5)
 
# 2. Define embedding network
class EmbeddingNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
 
    def forward(self, x):
        x = self.net(x)
        return x.squeeze(-1)
 
model = EmbeddingNet()
 
# 3. Compute embeddings
with torch.no_grad():
    support_tensor = torch.FloatTensor(support).unsqueeze(1)
    query_tensor = torch.FloatTensor(queries).unsqueeze(1)
 
    emb_support = model(support_tensor).numpy()
    emb_query = model(query_tensor).numpy()
 
# 4. Prototype classification (mean of support class embeddings)
proto_A = emb_support[support_labels == 0].mean(axis=0)
proto_B = emb_support[support_labels == 1].mean(axis=0)
 
prototypes = np.stack([proto_A, proto_B])
 
# 5. Classify queries by nearest prototype
dists = euclidean_distances(emb_query, prototypes)
preds = np.argmin(dists, axis=1)
 
# 6. Accuracy and plot
acc = (preds == query_labels).mean()
print(f"✅ Few-Shot Classification Accuracy: {acc:.2%}")
 
# Plot a few query predictions
plt.figure(figsize=(10, 4))
for i in range(5):
    plt.plot(queries[i], label=f"Query {i+1} - Predicted: {preds[i]}")
plt.title("Few-Shot Learning – Query Predictions")
plt.grid(True)
plt.legend()
plt.show()