"""
Project 151. Meta-learning implementation
Description:
Meta-learning, or "learning to learn," aims to train models that can quickly adapt to new tasks with minimal data. In this project, we implement Model-Agnostic Meta-Learning (MAML) — a popular meta-learning algorithm that learns initial parameters which can be fine-tuned rapidly on new tasks.

About:
🧠 What This Project Demonstrates:
Implements Model-Agnostic Meta-Learning (MAML) from scratch

Learns a good initialization that can quickly adapt to new tasks

Trains on sinusoidal regression tasks for clarity and visualization
"""

# Install if not already: pip install torch matplotlib
 
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
 
# Define a simple MLP for regression
class MAMLModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 40),
            nn.ReLU(),
            nn.Linear(40, 40),
            nn.ReLU(),
            nn.Linear(40, 1)
        )
 
    def forward(self, x):
        return self.net(x)
 
# Generate sinusoid regression tasks
def get_sinusoid_task(K=10):
    A = np.random.uniform(0.1, 5.0)
    phase = np.random.uniform(0, np.pi)
    x = np.random.uniform(-5.0, 5.0, size=(K, 1))
    y = A * np.sin(x + phase)
    return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
 
# MAML training
def train_maml(model, meta_lr=0.001, inner_lr=0.01, K=10, tasks=5, inner_steps=1):
    meta_optimizer = optim.Adam(model.parameters(), lr=meta_lr)
 
    for episode in range(200):
        meta_loss = 0.0
        for _ in range(tasks):
            x_train, y_train = get_sinusoid_task(K)
            x_val, y_val = get_sinusoid_task(K)
 
            # Clone model for task-specific adaptation
            fast_weights = list(model.parameters())
            for step in range(inner_steps):
                preds = model(x_train)
                loss = F.mse_loss(preds, y_train)
                grads = torch.autograd.grad(loss, model.parameters(), create_graph=True)
                fast_weights = [w - inner_lr * g for w, g in zip(fast_weights, grads)]
 
            # Evaluate adapted model on validation set
            adapted_preds = model_forward_with_weights(model, x_val, fast_weights)
            meta_loss += F.mse_loss(adapted_preds, y_val)
 
        # Meta-update
        meta_optimizer.zero_grad()
        meta_loss /= tasks
        meta_loss.backward()
        meta_optimizer.step()
 
        if episode % 20 == 0:
            print(f"📚 Episode {episode} — Meta Loss: {meta_loss.item():.4f}")
 
# Helper: forward pass with manual weights
def model_forward_with_weights(model, x, weights):
    x = F.linear(x, weights[0], weights[1])
    x = F.relu(x)
    x = F.linear(x, weights[2], weights[3])
    x = F.relu(x)
    x = F.linear(x, weights[4], weights[5])
    return x
 
# Train and test MAML
model = MAMLModel()
train_maml(model)
 
# Test on a new task
x_test, y_test = get_sinusoid_task(K=100)
with torch.no_grad():
    y_pred = model(x_test).numpy()
 
plt.plot(x_test.numpy(), y_test.numpy(), label="True")
plt.plot(x_test.numpy(), y_pred, label="MAML Prediction")
plt.legend()
plt.title("🎯 MAML on New Sinusoid Task")
plt.show()