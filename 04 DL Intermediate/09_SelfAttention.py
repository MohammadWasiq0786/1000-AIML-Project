"""
Project 129. Self-attention implementation
Description:
Self-attention is the foundation of the Transformer architecture. It allows each token in a sequence to attend to all other tokens, enabling the model to capture contextual relationships. In this project, we implement self-attention from scratch using PyTorch — where queries, keys, and values come from the same input.

About:
🧠 What This Project Demonstrates:
Implements the core of self-attention used in every transformer layer

Shows how every token attends to every other token in a sequence

Produces contextualized token embeddings with learned importance weights
"""

# Install if not already: pip install torch
 
import torch
import torch.nn as nn
import torch.nn.functional as F
 
class SelfAttention(nn.Module):
    def __init__(self, embed_dim):
        super(SelfAttention, self).__init__()
        self.embed_dim = embed_dim
        
        # Define linear layers for Query, Key, and Value
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        
        # Final output projection
        self.out = nn.Linear(embed_dim, embed_dim)
 
    def forward(self, x):
        # x: (batch_size, seq_len, embed_dim)
        Q = self.query(x)  # (B, L, D)
        K = self.key(x)
        V = self.value(x)
 
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.embed_dim, dtype=torch.float32))
        attn_weights = F.softmax(scores, dim=-1)
 
        # Attention output
        context = torch.matmul(attn_weights, V)
        output = self.out(context)
        return output, attn_weights
 
# Sample input: batch of 1 sequence, length 4, embedding dim 8
x = torch.rand(1, 4, 8)  # (B, L, D)
 
# Create self-attention layer and apply
self_attn = SelfAttention(embed_dim=8)
output, weights = self_attn(x)
 
# Display results
print("🔍 Attention Weights:\n", weights.squeeze(0).detach().numpy())
print("\n📤 Output Embeddings:\n", output.squeeze(0).detach().numpy())