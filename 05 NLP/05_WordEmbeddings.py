"""
Project 165. Word embeddings from scratch
Description:
Word embeddings represent words as dense vectors that capture semantic relationships, enabling machines to understand language contextually. In this project, we implement word2vec Skip-gram model from scratch using PyTorch to learn embeddings from a small text corpus.

About:
🧠 What This Project Demonstrates:
Builds a Skip-gram word2vec model from scratch using PyTorch

Learns to predict context words from a center word

Produces dense vector representations of words that capture meaning
"""

# Install if not already: pip install torch nltk
 
import torch
import torch.nn as nn
import torch.optim as optim
import nltk
from collections import Counter
import random
import numpy as np
 
nltk.download('punkt')
 
# Sample corpus
corpus = "The quick brown fox jumps over the lazy dog. The dog barked at the fox."
 
# Preprocess text
words = nltk.word_tokenize(corpus.lower())
vocab = list(set(words))
word2idx = {w: i for i, w in enumerate(vocab)}
idx2word = {i: w for w, i in word2idx.items()}
 
# Generate training data: (target, context) pairs using window size
window_size = 2
data = []
 
for i, target in enumerate(words):
    for j in range(-window_size, window_size + 1):
        if j == 0 or i + j < 0 or i + j >= len(words):
            continue
        context = words[i + j]
        data.append((word2idx[target], word2idx[context]))
 
# Define Skip-gram Model
class SkipGram(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(SkipGram, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.out = nn.Linear(embedding_dim, vocab_size)
 
    def forward(self, x):
        x = self.embedding(x)
        x = self.out(x)
        return x
 
# Hyperparameters
embedding_dim = 10
vocab_size = len(vocab)
model = SkipGram(vocab_size, embedding_dim)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)
 
# Training loop
for epoch in range(200):
    total_loss = 0
    for target, context in data:
        target_tensor = torch.tensor([target])
        context_tensor = torch.tensor([context])
 
        output = model(target_tensor)
        loss = loss_fn(output, context_tensor)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
 
    if epoch % 50 == 0:
        print(f"🧠 Epoch {epoch}, Loss: {total_loss:.4f}")
 
# Show word embeddings
print("\n🔍 Word Embeddings:")
for word in ["dog", "fox", "the", "lazy"]:
    idx = word2idx[word]
    embed = model.embedding.weight[idx].detach().numpy()
    print(f"{word}: {embed}")