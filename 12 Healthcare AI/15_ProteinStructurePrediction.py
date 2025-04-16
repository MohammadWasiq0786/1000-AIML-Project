"""
Project 455. Protein structure prediction
Description:
Protein structure prediction involves determining the 3D shape of a protein from its amino acid sequence. This structure determines its biological function. While models like AlphaFold2 are the state-of-the-art, we’ll implement a simplified version using secondary structure prediction (e.g., helix, sheet, coil) from protein sequences using a deep learning classifier.

About:
✅ What It Does:
Simulates protein sequences and predicts their secondary structure.

Uses a BiLSTM to model sequence patterns like helix, sheet, and coil.

Easily extendable to:

Include position-specific scoring matrices (PSSMs)

Work with real protein sequence databases

Move toward tertiary structure prediction with graph-based models or AlphaFold APIs

In real projects, you can use:

CB513 or CullPDB dataset

PSIPRED or AlphaFold tools for comparison
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import random
import numpy as np
 
# 1. Amino acids and structure labels
AMINO_ACIDS = 'ACDEFGHIKLMNPQRSTVWY'
STRUCTURES = ['H', 'E', 'C']  # Helix, Sheet, Coil
 
# 2. Encode amino acids
aa_to_idx = {aa: i for i, aa in enumerate(AMINO_ACIDS)}
structure_to_idx = {s: i for i, s in enumerate(STRUCTURES)}
 
# 3. Simulated dataset: sequence → secondary structure label
class ProteinDataset(Dataset):
    def __init__(self, num_samples=500, seq_len=50):
        self.data = []
        for _ in range(num_samples):
            seq = ''.join(random.choices(AMINO_ACIDS, k=seq_len))
            label = random.choices(STRUCTURES, k=seq_len)
            self.data.append((seq, label))
 
    def __len__(self):
        return len(self.data)
 
    def __getitem__(self, idx):
        seq, label = self.data[idx]
        x = torch.tensor([aa_to_idx[aa] for aa in seq], dtype=torch.long)
        y = torch.tensor([structure_to_idx[s] for s in label], dtype=torch.long)
        return x, y
 
# 4. BiLSTM model for sequence classification
class ProteinModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.bilstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
 
    def forward(self, x):
        x = self.embed(x)                   # [B, L, D]
        out, _ = self.bilstm(x)             # [B, L, 2H]
        logits = self.fc(out)               # [B, L, C]
        return logits
 
# 5. Setup
dataset = ProteinDataset()
loader = DataLoader(dataset, batch_size=16, shuffle=True)
model = ProteinModel(vocab_size=len(AMINO_ACIDS), embed_dim=32, hidden_dim=64, num_classes=len(STRUCTURES))
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()
 
# 6. Training loop
for epoch in range(1, 6):
    model.train()
    total, correct = 0, 0
    for seqs, labels in loader:
        optimizer.zero_grad()
        outputs = model(seqs)
        loss = loss_fn(outputs.view(-1, len(STRUCTURES)), labels.view(-1))
        loss.backward()
        optimizer.step()
 
        preds = outputs.argmax(-1)
        correct += (preds == labels).sum().item()
        total += labels.numel()
    print(f"Epoch {epoch}, Accuracy: {correct / total:.2f}")