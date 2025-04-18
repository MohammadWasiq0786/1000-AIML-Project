"""
Project 982: Bayesian Deep Learning Implementation
Description
Bayesian deep learning models incorporate uncertainty into predictions by treating weights as random variables with distributions, rather than fixed values. In this project, we will implement Bayesian neural networks to model uncertainty in deep learning predictions using variational inference.

Python Implementation with Comments (Bayesian Neural Networks using torch and Pyro)
We’ll use the Pyro library, which is built on PyTorch, to implement Bayesian Neural Networks. Pyro uses variational inference to approximate the posterior distribution of the model's weights.

First, install the necessary libraries:

pip install pyro-ppl torch torchvision

Key Concepts Covered:
Bayesian Neural Networks (BNNs): A neural network where the weights are treated as distributions (random variables) instead of fixed values.

Variational Inference: The method of approximating the true posterior distribution of weights using a variational distribution (guide).

Uncertainty Estimation: BNNs model uncertainty in predictions, which can be useful for applications requiring confidence measures in predictions.
"""


import torch
import torch.nn as nn
import pyro
import pyro.distributions as dist
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
 
# Define a Bayesian Neural Network model (a simple fully connected network)
class BayesianNN(nn.Module):
    def __init__(self):
        super(BayesianNN, self).__init__()
        self.fc1 = nn.Linear(784, 400)  # First fully connected layer
        self.fc2 = nn.Linear(400, 10)   # Output layer (10 classes for MNIST)
 
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
 
# Define a function to model the uncertainty in the weights
def model(x_data, y_data):
    # Priors for the weights
    fc1w_prior = dist.Normal(torch.zeros_like(model.fc1.weight), torch.ones_like(model.fc1.weight)).to_event('both')
    fc1b_prior = dist.Normal(torch.zeros_like(model.fc1.bias), torch.ones_like(model.fc1.bias)).to_event('both')
    fc2w_prior = dist.Normal(torch.zeros_like(model.fc2.weight), torch.ones_like(model.fc2.weight)).to_event('both')
    fc2b_prior = dist.Normal(torch.zeros_like(model.fc2.bias), torch.ones_like(model.fc2.bias)).to_event('both')
 
    # Priors for the weights of the network
    pyro.sample("fc1w", fc1w_prior)
    pyro.sample("fc1b", fc1b_prior)
    pyro.sample("fc2w", fc2w_prior)
    pyro.sample("fc2b", fc2b_prior)
    
    # Likelihood (Softmax likelihood)
    logits = model(x_data)
    pyro.sample("obs", dist.Categorical(logits=logits), obs=y_data)
 
# Define the guide (variational distribution)
def guide(x_data, y_data):
    # Variational distributions for the weights (approximating the posterior)
    fc1w_mean = pyro.param("fc1w_mean", torch.randn_like(model.fc1.weight))
    fc1w_scale = pyro.param("fc1w_scale", torch.ones_like(model.fc1.weight), constraint=torch.constraints.positive)
    fc1b_mean = pyro.param("fc1b_mean", torch.randn_like(model.fc1.bias))
    fc1b_scale = pyro.param("fc1b_scale", torch.ones_like(model.fc1.bias), constraint=torch.constraints.positive)
    fc2w_mean = pyro.param("fc2w_mean", torch.randn_like(model.fc2.weight))
    fc2w_scale = pyro.param("fc2w_scale", torch.ones_like(model.fc2.weight), constraint=torch.constraints.positive)
    fc2b_mean = pyro.param("fc2b_mean", torch.randn_like(model.fc2.bias))
    fc2b_scale = pyro.param("fc2b_scale", torch.ones_like(model.fc2.bias), constraint=torch.constraints.positive)
 
    # Use normal distributions for the guide (variational distribution)
    pyro.sample("fc1w", dist.Normal(fc1w_mean, fc1w_scale).to_event('both'))
    pyro.sample("fc1b", dist.Normal(fc1b_mean, fc1b_scale).to_event('both'))
    pyro.sample("fc2w", dist.Normal(fc2w_mean, fc2w_scale).to_event('both'))
    pyro.sample("fc2b", dist.Normal(fc2b_mean, fc2b_scale).to_event('both'))
 
# Load MNIST dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
 
# Initialize the model and optimizer
model = BayesianNN()
optimizer = optim.Adam(model.parameters(), lr=0.001)
 
# Define the loss function (variational loss)
def loss_fn(model, guide, x_data, y_data):
    return pyro.infer.Trace_ELBO().differentiable_loss(model, guide, x_data, y_data)
 
# Train the Bayesian Neural Network
for epoch in range(5):  # Training for 5 epochs
    model.train()
    total_loss = 0
    for data, target in train_loader:
        data = data.view(-1, 784)  # Flatten MNIST images to vectors
 
        # Run the model and guide
        optimizer.zero_grad()
        loss = loss_fn(model, guide, data, target)
        loss.backward()
        optimizer.step()
 
        total_loss += loss.item()
 
    print(f"Epoch {epoch+1}, Loss: {total_loss / len(train_loader)}")
 
# Test the model on the test set
model.eval()
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
 
correct = 0
total = 0
with torch.no_grad():
    for data, target in test_loader:
        data = data.view(-1, 784)
        output = model(data)
        _, predicted = torch.max(output, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()
 
print(f"Accuracy on test data: {100 * correct / total:.2f}%")