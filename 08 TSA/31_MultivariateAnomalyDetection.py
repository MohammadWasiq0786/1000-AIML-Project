"""
Project 311. Multivariate anomaly detection
Description:
Multivariate anomaly detection identifies unusual data points in datasets with multiple variables (e.g., temperature, pressure, vibration). This is crucial for:

Sensor fusion systems

Industrial equipment monitoring

Financial fraud detection

We’ll use an autoencoder neural network to reconstruct input data and flag data points with high reconstruction error as anomalies.

About:
✅ What It Does:
Simulates 3D normal data with injected outliers

Trains an autoencoder to learn typical patterns

Flags anomalies based on high reconstruction loss

Visualizes the error trend and threshold
"""

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
 
# 1. Simulate multivariate normal data with injected anomalies
np.random.seed(42)
n_samples = 1000
n_anomalies = 30
 
# Normal multivariate data (3 features)
normal_data = np.random.normal(0, 1, (n_samples, 3))
 
# Inject anomalies (random spikes)
anomalies = np.random.uniform(-6, 6, (n_anomalies, 3))
data = np.vstack([normal_data, anomalies])
labels = np.array([0] * n_samples + [1] * n_anomalies)  # 0: normal, 1: anomaly
 
# Shuffle
idx = np.random.permutation(len(data))
data, labels = data[idx], labels[idx]
 
# Normalize
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)
data_tensor = torch.FloatTensor(data_scaled)
 
# 2. Autoencoder definition
class AE(nn.Module):
    def __init__(self, input_dim=3):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 8),
            nn.ReLU(),
            nn.Linear(8, 3)
        )
        self.decoder = nn.Sequential(
            nn.Linear(3, 8),
            nn.ReLU(),
            nn.Linear(8, input_dim)
        )
 
    def forward(self, x):
        return self.decoder(self.encoder(x))
 
model = AE()
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
loss_fn = nn.MSELoss()
 
# 3. Train autoencoder
for epoch in range(20):
    output = model(data_tensor)
    loss = loss_fn(output, data_tensor)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item():.6f}")
 
# 4. Compute reconstruction error
model.eval()
with torch.no_grad():
    recon = model(data_tensor)
    errors = torch.mean((recon - data_tensor) ** 2, dim=1).numpy()
 
# 5. Thresholding
threshold = np.percentile(errors, 95)  # Top 5% as anomalies
predicted_labels = (errors > threshold).astype(int)
 
# 6. Plot reconstruction error
plt.figure(figsize=(10, 4))
plt.plot(errors, label="Reconstruction Error")
plt.axhline(threshold, color='red', linestyle='--', label="Anomaly Threshold")
plt.title("Multivariate Anomaly Detection – Autoencoder")
plt.xlabel("Sample")
plt.ylabel("Error")
plt.legend()
plt.grid(True)
plt.show()
 
# Accuracy check
accuracy = (predicted_labels == labels).mean()
print(f"✅ Detection Accuracy: {accuracy:.2%}")