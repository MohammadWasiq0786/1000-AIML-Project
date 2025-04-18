"""
Project 981: Neural Ordinary Differential Equations (Neural ODEs)
Description
Neural Ordinary Differential Equations (ODEs) are a type of neural network architecture where the hidden states are treated as the solution to an ODE. Instead of using discrete layers, a Neural ODE integrates a differential equation over time to model the hidden states. This approach is particularly useful for continuous-time models and offers a more memory-efficient way of learning.

Key Concepts Covered:
Neural ODEs: Neural networks where the hidden states are modeled as the solution to a differential equation.

ODE Solver: The torchdiffeq library is used to solve the ODE efficiently during training and evaluation.

Continuous-Time Models: Neural ODEs are designed to work with continuous-time models, offering a memory-efficient alternative to traditional discrete-layer networks.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchdiffeq import odeint_adjoint as odeint
import numpy as np
import matplotlib.pyplot as plt
 
# Define the Neural ODE model (learns the derivative of the hidden state)
class ODEFunc(nn.Module):
    def __init__(self):
        super(ODEFunc, self).__init__()
        self.fc1 = nn.Linear(1, 50)
        self.fc2 = nn.Linear(50, 1)
 
    def forward(self, t, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
 
# Define the Neural ODE model that uses the ODEFunc
class NeuralODE(nn.Module):
    def __init__(self, ode_func):
        super(NeuralODE, self).__init__()
        self.ode_func = ode_func
 
    def forward(self, x0, t):
        # Solve the ODE from initial state x0 over time t
        out = odeint(self.ode_func, x0, t)
        return out
 
# Define a simple dataset for training
def generate_data():
    # True function is a simple sine function with noise
    t = torch.linspace(0., 25., 100)
    y = torch.sin(t) + 0.1 * torch.randn_like(t)
    return t, y
 
# Generate data
t, y = generate_data()
 
# Initialize the ODE function and the model
ode_func = ODEFunc()
model = NeuralODE(ode_func)
 
# Initial condition (start from 0)
x0 = torch.tensor([[0.0]])
 
# Time points for ODE solver
t_points = torch.linspace(0., 25., 100)
 
# Loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)
 
# Training loop
for epoch in range(500):
    model.train()
    
    # Zero gradients
    optimizer.zero_grad()
 
    # Forward pass through the Neural ODE model
    pred_y = model(x0, t_points)
 
    # Compute the loss
    loss = criterion(pred_y.squeeze(), y)
    
    # Backward pass
    loss.backward()
    optimizer.step()
 
    if epoch % 50 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
 
# Plot the results
model.eval()
with torch.no_grad():
    pred_y = model(x0, t_points)
 
plt.plot(t_points.numpy(), y.numpy(), label='True Function (Sine)')
plt.plot(t_points.numpy(), pred_y.squeeze().numpy(), label='Predicted by Neural ODE')
plt.legend()
plt.show()
