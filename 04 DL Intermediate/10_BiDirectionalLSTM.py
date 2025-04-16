"""
Project 130. Bi-directional LSTM for NLP
Description:
Bi-directional LSTMs (BiLSTMs) process input sequences in both forward and backward directions, capturing context from both past and future tokens. This is especially useful for tasks like named entity recognition, text classification, and part-of-speech tagging. In this project, we implement a BiLSTM-based text classification model using PyTorch.

About:
🧠 What This Project Demonstrates:
Builds a Bi-directional LSTM model using PyTorch

Applies to binary sentiment classification on the IMDB dataset

Uses embedding layer, BiLSTM, and mean pooling for sequence encoding
"""

# Install if not already: pip install torch torchvision torchtext
 
import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.datasets import IMDB
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import random
 
# Set random seed for reproducibility
torch.manual_seed(0)
 
# Load IMDB dataset (for binary classification)
train_iter = IMDB(split='train')
 
# Tokenization and vocabulary
tokenizer = get_tokenizer('basic_english')
def yield_tokens(data_iter):
    for label, text in data_iter:
        yield tokenizer(text)
 
vocab = build_vocab_from_iterator(yield_tokens(train_iter), specials=["<pad>"])
vocab.set_default_index(vocab["<pad>"])
 
# Prepare data
def collate_batch(batch):
    text_list, label_list = [], []
    for label, text in batch:
        tokens = vocab(tokenizer(text))
        text_list.append(torch.tensor(tokens, dtype=torch.long))
        label_list.append(torch.tensor(1 if label == 'pos' else 0, dtype=torch.long))
    return pad_sequence(text_list, batch_first=True), torch.tensor(label_list)
 
train_iter = IMDB(split='train')
train_dataloader = DataLoader(list(train_iter), batch_size=16, shuffle=True, collate_fn=collate_batch)
 
# Define BiLSTM model
class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim):
        super(BiLSTMClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=vocab["<pad>"])
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(0.3)
 
    def forward(self, x):
        embedded = self.embedding(x)  # (B, L, D)
        lstm_out, _ = self.lstm(embedded)  # (B, L, 2H)
        pooled = torch.mean(lstm_out, dim=1)  # Average pooling
        return self.fc(self.dropout(pooled))
 
# Model, loss, optimizer
model = BiLSTMClassifier(vocab_size=len(vocab), embed_dim=100, hidden_dim=128, output_dim=2)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
 
# Simple training loop (1 epoch for demo)
model.train()
for batch in train_dataloader:
    inputs, targets = batch
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()
    break  # Remove break to train on full data
print("✅ Trained one batch successfully!")