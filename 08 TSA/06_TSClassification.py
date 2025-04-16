"""
Project 286. Time series classification
Description:
Time Series Classification involves predicting a label or category based on the characteristics of an entire sequence. This is widely used in activity recognition, sensor data, ECG signal analysis, and more.

In this project, we simulate time series classification using the UCR Time Series Classification Dataset. We’ll use a 1D CNN (convolutional neural network) to classify sequences.

About:
✅ What It Does:
Loads a pre-processed time series classification dataset

Uses 1D convolutional layers to extract temporal features

Trains a model to classify full sequences

Evaluates performance using accuracy score
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score
 
# Load sample dataset: ECGFiveDays from UCR archive (already preprocessed)
url = "https://raw.githubusercontent.com/hfawaz/cd-diagram/master/FordA/FordA_TRAIN.tsv"
df = pd.read_csv(url, sep="\t", header=None)
X = df.iloc[:, 1:].values
y = df.iloc[:, 0].values
y = LabelEncoder().fit_transform(y)
 
# Normalize and reshape
X = StandardScaler().fit_transform(X)
X = X[:, :, np.newaxis]  # Add channel dimension for CNN
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
 
X_train, X_test = torch.FloatTensor(X_train), torch.FloatTensor(X_test)
y_train, y_test = torch.LongTensor(y_train), torch.LongTensor(y_test)
 
# Define 1D CNN model
class CNN1D(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=5)
        self.pool = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3)
        self.fc1 = nn.Linear(64 * 22, 64)
        self.fc2 = nn.Linear(64, 2)
 
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)
 
model = CNN1D()
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()
 
# Training
epochs = 10
for epoch in range(epochs):
    model.train()
    output = model(X_train.permute(0, 2, 1))  # (B, C, L)
    loss = loss_fn(output, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
 
# Evaluation
model.eval()
with torch.no_grad():
    predictions = model(X_test.permute(0, 2, 1)).argmax(dim=1)
    acc = accuracy_score(y_test, predictions)
    print(f"Test Accuracy: {acc:.2%}")