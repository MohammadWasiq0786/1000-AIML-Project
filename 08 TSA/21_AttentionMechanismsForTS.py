"""
Project 301. Attention mechanisms for time series
Description:
Attention mechanisms let models dynamically focus on the most relevant parts of the input when making predictions. In time series, attention helps:

Capture important time steps

Learn variable-length dependencies

Improve interpretability

We’ll build a basic attention layer on top of an RNN to forecast a value based on a weighted sum of relevant hidden states.

About:
✅ What It Does:
Builds a GRU + attention layer

Learns to focus on the most important time steps in the input sequence

Produces a weighted context vector for final prediction

Improves modeling of complex dependencies in time series
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
 
# 1. Simulate time series where output depends on weighted past
np.random.seed(42)
n = 500
x = np.sin(np.linspace(0, 50, n)) + np.random.normal(0, 0.1, n)
y = np.array([0.3*x[i-5] + 0.7*x[i-2] for i in range(5, n)])
 
x = x[5:]
seq_len = 20
 
# Create sequences
def create_sequences(data, targets, seq_len):
    X, Y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i+seq_len])
        Y.append(targets[i+seq_len])
    return torch.FloatTensor(X).unsqueeze(-1), torch.FloatTensor(Y)
 
X, Y = create_sequences(x, y, seq_len)
dataset = TensorDataset(X, Y)
loader = DataLoader(dataset, batch_size=32, shuffle=True)
 
# 2. RNN + Attention model
class AttentionRNN(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=32):
        super().__init__()
        self.rnn = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.attn = nn.Linear(hidden_dim, 1)
        self.fc = nn.Linear(hidden_dim, 1)
 
    def forward(self, x):
        rnn_out, _ = self.rnn(x)  # [batch, seq_len, hidden]
        attn_weights = torch.softmax(self.attn(rnn_out).squeeze(-1), dim=1)  # [batch, seq_len]
        context = torch.sum(rnn_out * attn_weights.unsqueeze(-1), dim=1)     # [batch, hidden]
        return self.fc(context).squeeze()
 
model = AttentionRNN()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()
 
# 3. Train the model
for epoch in range(20):
    for batch_x, batch_y in loader:
        preds = model(batch_x)
        loss = loss_fn(preds, batch_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item():.6f}")
 
# 4. Evaluate
model.eval()
with torch.no_grad():
    predictions = model(X).numpy()
 
plt.figure(figsize=(10, 4))
plt.plot(Y.numpy(), label="True")
plt.plot(predictions, label="Attention Prediction", alpha=0.7)
plt.title("RNN with Attention – Time Series Forecasting")
plt.legend()
plt.grid(True)
plt.show()