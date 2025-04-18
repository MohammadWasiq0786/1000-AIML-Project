"""
Project 1000: Hybrid Deep Learning Architecture
Description
Hybrid deep learning architectures combine different neural network types, each specializing in a certain aspect of the learning task. This can involve combining Convolutional Neural Networks (CNNs) for feature extraction, Recurrent Neural Networks (RNNs) for sequence learning, or Transformer-based models for attention-based learning. In this project, we will build a hybrid architecture that combines a CNN for feature extraction and a RNN (LSTM) for sequence prediction.

Key Concepts Covered:
Hybrid Deep Learning Models: Combining different types of neural networks, such as CNNs for spatial feature extraction and RNNs (LSTMs) for sequence learning.

CNN: Convolutional Neural Networks are used for extracting hierarchical features from image or time-series data.

LSTM: Long Short-Term Memory (LSTM) networks are used for learning temporal dependencies and sequences.

End-to-End Learning: The model learns spatial features through CNNs and temporal features through LSTMs, combining both into a robust architecture.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np
 
# Define the hybrid CNN-LSTM model
class CNN_LSTM(nn.Module):
    def __init__(self, input_channels, output_size, cnn_filters=32, lstm_units=64):
        super(CNN_LSTM, self).__init__()
        
        # Convolutional Layer for feature extraction
        self.conv1 = nn.Conv2d(input_channels, cnn_filters, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # LSTM Layer for sequence learning
        self.lstm = nn.LSTM(input_size=cnn_filters * 8 * 8, hidden_size=lstm_units, batch_first=True)
        
        # Fully Connected Layer for final classification or prediction
        self.fc = nn.Linear(lstm_units, output_size)
 
    def forward(self, x):
        # Apply convolutional layers to extract spatial features
        x = self.pool(torch.relu(self.conv1(x)))
        x = x.view(x.size(0), -1, x.size(1) * x.size(2))  # Flatten to pass to LSTM
        
        # Apply LSTM to capture temporal patterns
        lstm_out, (hn, cn) = self.lstm(x)
        
        # Use the final hidden state to predict output
        out = self.fc(hn[-1])  # Taking the last hidden state
        
        return out
 
# Dataset creation (For demonstration, we will use random data)
class RandomTimeSeriesDataset(Dataset):
    def __init__(self, num_samples=1000, input_channels=1, sequence_length=30):
        self.num_samples = num_samples
        self.input_channels = input_channels
        self.sequence_length = sequence_length
        self.data = torch.randn(num_samples, input_channels, 30, 30)  # Random 30x30 images as sequences
        self.labels = torch.randint(0, 2, (num_samples,))  # Random binary labels (0 or 1)
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
 
# Hyperparameters
input_channels = 1  # Grayscale images
output_size = 2  # Binary classification
batch_size = 32
learning_rate = 0.001
epochs = 5
 
# Create the dataset and DataLoader
train_dataset = RandomTimeSeriesDataset(num_samples=1000, input_channels=input_channels)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
 
# Initialize the model, loss function, and optimizer
model = CNN_LSTM(input_channels=input_channels, output_size=output_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
 
# Training loop
for epoch in range(epochs):
    model.train()
    total_loss = 0
    correct_predictions = 0
    
    for data, labels in train_loader:
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(data)
        
        # Compute the loss
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # Calculate accuracy
        _, predicted = torch.max(outputs, 1)
        correct_predictions += (predicted == labels).sum().item()
    
    accuracy = 100 * correct_predictions / len(train_loader.dataset)
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(train_loader):.4f}, Accuracy: {accuracy:.2f}%")
 
# Save the trained model
torch.save(model.state_dict(), "cnn_lstm_model.pth")