"""
Project 456. Genomic sequence analysis
Description:
Genomic sequence analysis involves studying DNA/RNA sequences to uncover insights about gene functions, mutations, or regulatory patterns. In this project, we’ll build a simple deep learning model to classify DNA sequences (e.g., promoter vs non-promoter) using one-hot encoding and a 1D CNN.

About:
✅ What It Does:
Simulates DNA sequences and labels them as promoter/non-promoter.

Uses a 1D CNN with one-hot encoding to learn sequence patterns.

Can be extended to:

Detect splice sites, mutations, or motifs

Use k-mer embedding or transformers for long DNA reads

Apply to variant classification in genomics pipelines

For real-world data, you can explore:

UCI Promoter Gene Dataset

ENCODE Project

NCBI GenBank Sequences
"""


import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import random
import numpy as np
 
# 1. DNA base encoding
DNA_BASES = ['A', 'C', 'G', 'T']
base_to_idx = {base: i for i, base in enumerate(DNA_BASES)}
 
# 2. One-hot encode DNA sequence
def one_hot_encode(seq, seq_len=50):
    encoding = np.zeros((4, seq_len), dtype=np.float32)
    for i, base in enumerate(seq):
        if base in base_to_idx:
            encoding[base_to_idx[base], i] = 1.0
    return encoding
 
# 3. Simulated dataset (promoter vs non-promoter)
class DNADataset(Dataset):
    def __init__(self, num_samples=1000, seq_len=50):
        self.sequences = []
        self.labels = []
        for _ in range(num_samples):
            if random.random() < 0.5:
                seq = ''.join(random.choices('ACGT', k=seq_len))
                label = 0  # non-promoter
            else:
                seq = ''.join(random.choices('TATAAAACGT', k=seq_len))
                label = 1  # simulated promoter with TATA box
            self.sequences.append(one_hot_encode(seq))
            self.labels.append(label)
 
    def __len__(self):
        return len(self.sequences)
 
    def __getitem__(self, idx):
        return torch.tensor(self.sequences[idx]), self.labels[idx]
 
# 4. Define 1D CNN for sequence classification
class DNAClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(4, 32, kernel_size=5, padding=2)
        self.pool = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.fc = nn.Linear(64 * 25, 2)  # 50 -> 25 after pooling
 
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))  # [B, 32, 25]
        x = self.pool(torch.relu(self.conv2(x)))  # [B, 64, 12]
        x = x.view(x.size(0), -1)
        return self.fc(x)
 
# 5. Setup
dataset = DNADataset()
loader = DataLoader(dataset, batch_size=16, shuffle=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DNAClassifier().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()
 
# 6. Training loop
for epoch in range(1, 6):
    model.train()
    total, correct = 0, 0
    for seqs, labels in loader:
        seqs, labels = seqs.to(device), torch.tensor(labels).to(device)
        outputs = model(seqs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
 
        preds = outputs.argmax(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    print(f"Epoch {epoch}, Accuracy: {correct / total:.2f}")