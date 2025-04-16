"""
Project 128. Attention mechanism implementation
Description:
The attention mechanism allows neural networks to focus on specific parts of the input when generating each part of the output. It is the core innovation behind transformers, enabling models to weigh the importance of different tokens. In this project, we implement a basic scaled dot-product attention function in Python using PyTorch.

About:
🧠 What This Project Demonstrates:
Implements core attention mechanism used in transformers

Shows how queries, keys, and values interact through dot-products

Computes attention weights and uses them to produce output embeddings
"""

# Install if not already: pip install torch
 
import torch
import torch.nn.functional as F
 
# Simulated input: queries, keys, and values (batch_size=1, seq_len=5, dim=4)
Q = torch.tensor([[[0.1, 0.2, 0.3, 0.4],
                   [0.2, 0.1, 0.0, 0.1],
                   [0.5, 0.2, 0.3, 0.1],
                   [0.0, 0.3, 0.3, 0.7],
                   [0.9, 0.2, 0.4, 0.0]]], dtype=torch.float32)
 
K = Q.clone()  # In self-attention, Q=K=V
V = Q.clone()
 
def scaled_dot_product_attention(Q, K, V):
    # Step 1: Calculate raw attention scores by dot product
    scores = torch.matmul(Q, K.transpose(-2, -1))  # Shape: (batch, seq_len, seq_len)
    
    # Step 2: Scale scores
    d_k = Q.size(-1)
    scaled_scores = scores / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))
 
    # Step 3: Apply softmax to get attention weights
    attention_weights = F.softmax(scaled_scores, dim=-1)
 
    # Step 4: Weighted sum of values
    output = torch.matmul(attention_weights, V)
    return output, attention_weights
 
# Run attention
output, weights = scaled_dot_product_attention(Q, K, V)
 
# Display results
print("🎯 Attention Weights:\n", weights[0].detach().numpy())
print("\n📤 Output Representation:\n", output[0].detach().numpy())