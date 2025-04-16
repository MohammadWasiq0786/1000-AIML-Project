"""
Project 303. Neural ordinary differential equations
Description:
Neural ODEs model time series using continuous-time dynamics instead of discrete layers. They define the derivative of the hidden state with a neural network and solve it using an ODE solver.

This approach is great for:

Irregular time series

Physics-informed modeling

Data-efficient learning

We'll build a simple Neural ODE model using the torchdiffeq library to predict a smooth time series.

About:
✅ What It Does:
Uses a neural network to define the dynamics of the system

Solves the system using a differentiable ODE solver (odeint)

Smooths and predicts a noisy sine wave as a continuous trajectory

Perfect for irregular or sparse time series modeling
"""

# Install if needed:
# pip install torchdiffeq
 
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torchdiffeq import odeint
 
# 1. Generate noisy sine wave
np.random.seed(0)
t = torch.linspace(0, 10, 100)
y_true = torch.sin(t)
y_noisy = y_true + 0.1 * torch.randn_like(t)
 
# 2. Define neural ODE function (dy/dt = f(t, y))
class ODEFunc(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )
 
    def forward(self, t, y):
        return self.net(y)
 
# 3. Define ODE block
class NeuralODE(nn.Module):
    def __init__(self, ode_func):
        super().__init__()
        self.ode_func = ode_func
 
    def forward(self, y0, t):
        return odeint(self.ode_func, y0, t)
 
# 4. Initialize model
func = ODEFunc()
model = NeuralODE(func)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.MSELoss()
 
# Initial condition: noisy first point
y0 = y_noisy[0].unsqueeze(0)
 
# 5. Train Neural ODE
for epoch in range(200):
    optimizer.zero_grad()
    pred_y = model(y0, t).squeeze()
    loss = loss_fn(pred_y, y_noisy)
    loss.backward()
    optimizer.step()
    if epoch % 20 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.5f}")
 
# 6. Plot predictions
plt.figure(figsize=(10, 4))
plt.plot(t, y_noisy, label="Noisy Observation", alpha=0.6)
plt.plot(t, model(y0, t).detach().squeeze(), label="Neural ODE Prediction", linewidth=2)
plt.plot(t, y_true, label="True Signal", linestyle="--")
plt.title("Neural ODE – Time Series Smoothing")
plt.legend()
plt.grid(True)
plt.show()