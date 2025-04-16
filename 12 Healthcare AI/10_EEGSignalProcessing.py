"""
Project 450. EEG signal processing
Description:
EEG (Electroencephalogram) signals capture brain activity through electrical patterns. Analyzing EEG is critical for diagnosing epilepsy, sleep disorders, or for brain-computer interfaces (BCI). In this project, we’ll simulate EEG data and use a 1D CNN to classify brain signals as normal or epileptic.

✅ What It Does:
Simulates multichannel EEG signals.

Trains a 1D CNN to classify brainwave patterns.

Can be extended to:

Work with real EEG recordings

Perform event detection, e.g., seizures

Enable neurofeedback systems or brain-computer interfaces

Real-world datasets you can use later:

TUH EEG Seizure Corpus

CHB-MIT Scalp EEG Dataset
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
 
# 1. Simulate EEG dataset (binary classification)
class EEGDataset(Dataset):
    def __init__(self, num_samples=200, channels=14, seq_len=128):
        self.data = []
        self.labels = []
        for _ in range(num_samples):
            if random.random() < 0.5:  # normal
                signal = np.random.normal(0, 0.1, (channels, seq_len))
                label = 0
            else:  # epileptic or abnormal
                signal = np.random.normal(1, 0.5, (channels, seq_len))
                label = 1
            self.data.append(torch.tensor(signal, dtype=torch.float32))
            self.labels.append(label)
 
    def __len__(self):
        return len(self.data)
 
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
 
# 2. 1D CNN model for EEG signals
class EEGNet(nn.Module):
    def __init__(self, channels=14, seq_len=128):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool1d(2)
        self.fc1 = nn.Linear(64 * (seq_len // 4), 64)
        self.fc2 = nn.Linear(64, 2)
 
    def forward(self, x):
        x = self.pool1(torch.relu(self.conv1(x)))  # [B, 32, L/2]
        x = self.pool2(torch.relu(self.conv2(x)))  # [B, 64, L/4]
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)
 
# 3. Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset = EEGDataset()
loader = DataLoader(dataset, batch_size=16, shuffle=True)
model = EEGNet().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()
 
# 4. Training loop
for epoch in range(1, 6):
    model.train()
    total, correct = 0, 0
    for signals, labels in loader:
        signals, labels = signals.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(signals)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        total += labels.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
    print(f"Epoch {epoch}, Accuracy: {correct / total:.2f}")