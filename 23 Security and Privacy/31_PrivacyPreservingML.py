"""
Project 911. Privacy-Preserving Machine Learning

Privacy-preserving machine learning enables model training and inference without exposing sensitive data. In this project, we simulate training a model on private data using differential privacy via PyTorch and Opacus.

What This Demonstrates:
Differential privacy is achieved by adding noise + clipping gradients

You get a measurable privacy budget: ε (epsilon) and δ (delta)

Real use cases: healthcare, finance, personalized AI with sensitive user data

🔐 You can also explore:

Federated learning (no data leaves the device)

Secure MPC and Homomorphic Encryption

DP-SGD for other models like transformers or CNNs
"""

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from opacus import PrivacyEngine
 
# Simulated private dataset (e.g., medical records)
X = torch.randn(100, 10)         # 100 samples, 10 features
y = (X[:, 0] > 0).float()        # binary labels based on first feature
 
# Create DataLoader
dataset = TensorDataset(X, y)
data_loader = DataLoader(dataset, batch_size=16, shuffle=True)
 
# Define simple model
model = nn.Sequential(
    nn.Linear(10, 1),
    nn.Sigmoid()
)
 
# Loss and optimizer
criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)
 
# Attach PrivacyEngine
privacy_engine = PrivacyEngine()
model, optimizer, data_loader = privacy_engine.make_private(
    module=model,
    optimizer=optimizer,
    data_loader=data_loader,
    noise_multiplier=1.0,      # controls noise level
    max_grad_norm=1.0          # clips gradients for DP
)
 
# Training loop
model.train()
for epoch in range(5):
    for batch_x, batch_y in data_loader:
        optimizer.zero_grad()
        pred = model(batch_x).squeeze()
        loss = criterion(pred, batch_y)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
 
# Show DP accounting
epsilon, best_alpha = privacy_engine.accountant.get_privacy_spent(delta=1e-5)
print(f"\n✅ Training complete with (ε = {epsilon:.2f}, δ = 1e-5) at α = {best_alpha}")