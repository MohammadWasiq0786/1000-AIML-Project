"""
Project 983: Probabilistic Programming Implementation
Description
Probabilistic programming is a technique where models are specified using probabilistic terms, enabling the integration of uncertainty into machine learning models. In this project, we will implement a simple probabilistic program using the Pyro library, a probabilistic programming framework built on top of PyTorch.

Key Concepts Covered:
Probabilistic Programming: Modeling uncertain parameters as random variables and defining priors and likelihoods in the program.

Variational Inference: Using Pyro's variational inference tools to approximate the posterior distribution of model parameters.

Bayesian Linear Regression: A linear regression model where both the slope and the intercept are treated as probabilistic quantities, and the model’s uncertainty is quantified.
"""

import torch
import pyro
import pyro.distributions as dist
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
 
# Define the linear regression model with probabilistic parameters
def model(x_data, y_data):
    # Define priors for the parameters (slope and intercept)
    slope = pyro.sample("slope", dist.Normal(0., 1.))  # Prior for slope (mean 0, std 1)
    intercept = pyro.sample("intercept", dist.Normal(0., 1.))  # Prior for intercept
 
    # Define a prior for the noise (standard deviation of the residuals)
    noise = pyro.sample("noise", dist.HalfNormal(1.))  # Noise (sigma)
    
    # Define the linear model: y = slope * x + intercept
    y_hat = slope * x_data + intercept
    
    # Likelihood: Assuming Gaussian noise for the observations
    with pyro.plate("data", len(x_data)):
        pyro.sample("obs", dist.Normal(y_hat, noise), obs=y_data)  # Likelihood with observed data
 
# Define the guide (variational distribution)
def guide(x_data, y_data):
    # Variational distributions for slope, intercept, and noise
    slope_loc = pyro.param("slope_loc", torch.tensor(0.))
    slope_scale = pyro.param("slope_scale", torch.tensor(1.), constraint=torch.constraints.positive)
    
    intercept_loc = pyro.param("intercept_loc", torch.tensor(0.))
    intercept_scale = pyro.param("intercept_scale", torch.tensor(1.), constraint=torch.constraints.positive)
    
    noise_loc = pyro.param("noise_loc", torch.tensor(1.), constraint=torch.constraints.positive)
 
    # Sample from the variational distribution
    slope = pyro.sample("slope", dist.Normal(slope_loc, slope_scale))
    intercept = pyro.sample("intercept", dist.Normal(intercept_loc, intercept_scale))
    noise = pyro.sample("noise", dist.HalfNormal(noise_loc))
 
# Generate synthetic data (linear regression with some noise)
np.random.seed(0)
x_data = np.linspace(0, 10, 100)
y_data = 2 * x_data + 1 + np.random.normal(0, 1, 100)
 
# Convert data to tensors
x_data = torch.tensor(x_data, dtype=torch.float)
y_data = torch.tensor(y_data, dtype=torch.float)
 
# Prepare DataLoader
data = TensorDataset(x_data, y_data)
data_loader = DataLoader(data, batch_size=32, shuffle=True)
 
# Define the optimizer
optimizer = optim.Adam([{'params': pyro.get_param_store().values()}], lr=0.01)
 
# Set the number of iterations for training
num_iterations = 1000
 
# Run inference using variational inference
for iteration in range(num_iterations):
    optimizer.zero_grad()
    
    # Perform one step of inference
    loss = pyro.infer.Trace_ELBO().differentiable_loss(model, guide, x_data, y_data)
    loss.backward()
    optimizer.step()
 
    if iteration % 100 == 0:
        print(f"Iteration {iteration}, Loss: {loss.item()}")
 
# Get the learned parameters
slope_loc = pyro.param("slope_loc").item()
intercept_loc = pyro.param("intercept_loc").item()
noise_loc = pyro.param("noise_loc").item()
 
print(f"Learned Slope: {slope_loc:.2f}, Intercept: {intercept_loc:.2f}, Noise: {noise_loc:.2f}")
 
# Plot the results
plt.scatter(x_data.numpy(), y_data.numpy(), label="Observed data")
plt.plot(x_data.numpy(), slope_loc * x_data.numpy() + intercept_loc, color='red', label="Fitted line")
plt.legend()
plt.show()