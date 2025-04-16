"""
Project 449. ECG analysis system
Description:
An ECG (Electrocardiogram) Analysis System interprets electrical signals from the heart to detect conditions like arrhythmias, atrial fibrillation, or myocardial infarction. In this project, we’ll build a simple 1D CNN to classify ECG signals as normal or abnormal, based on waveform patterns.

About:
✅ What It Does:
Simulates ECG signal data (normal vs abnormal).

Trains a 1D CNN to classify based on signal patterns.

Can be extended to:

Multi-class arrhythmia classification

Integration with real ECG datasets (PhysioNet)

Wavelet transforms or signal augmentation

For production use, consider real datasets like:

MIT-BIH Arrhythmia Database (via PhysioNet)

PTB-XL ECG Dataset
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
 
# 1. Simulate ECG signal dataset (1D time series data)
class ECGDataset(Dataset):
    def __init__(self, num_samples=200, seq_length=250):
        self.data = []
        self.labels = []
        for _ in range(num_samples):
            if random.random() < 0.5:  # normal
                signal = np.sin(np.linspace(0, 8 * np.pi, seq_length)) + np.random.normal(0, 0.1, seq_length)
                label = 0
            else:  # abnormal
                signal = np.random.normal(0, 1, seq_length)
                label = 1
            self.data.append(torch.tensor(signal, dtype=torch.float32))
            self.labels.append(label)
 
    def __len__(self):
        return len(self.data)
 
    def __getitem__(self, idx):
        signal = self.data[idx].unsqueeze(0)  # add channel dimension
        label = self.labels[idx]
        return signal, label
 
# 2. Define 1D CNN for ECG
class ECGCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=5, stride=1, padding=2)
        self.pool = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=5, stride=1, padding=2)
        self.fc1 = nn.Linear(32 * 62, 64)
        self.fc2 = nn.Linear(64, 2)
 
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # [B, 16, 125]
        x = self.pool(F.relu(self.conv2(x)))  # [B, 32, 62]
        x = x.view(x.size(0), -1)            # Flatten
        x = F.relu(self.fc1(x))
        return self.fc2(x)
 
# 3. Setup training
train_data = ECGDataset()
train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ECGCNN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()
 
# 4. Train model
for epoch in range(1, 6):
    model.train()
    running_loss, correct, total = 0, 0, 0
    for signals, labels in train_loader:
        signals, labels = signals.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(signals)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)
    acc = correct / total
    print(f"Epoch {epoch}, Loss: {running_loss:.4f}, Accuracy: {acc:.2f}")