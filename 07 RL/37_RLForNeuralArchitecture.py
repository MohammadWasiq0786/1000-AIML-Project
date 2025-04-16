"""
Project 277. RL for neural architecture search
Description:
Neural Architecture Search (NAS) is the process of using reinforcement learning to automatically discover optimal neural network architectures. An RL controller (often an RNN) generates sequences representing different architectures. Each sequence is trained and evaluated, and the controller gets a reward based on its performance (e.g., accuracy).

In this project, we implement a toy NAS framework, where an RL agent selects layer configurations for a small feedforward network. The agent is rewarded based on the validation accuracy of the resulting model trained on a subset of MNIST.

About:
✅ What It Does:
Controller generates architectures (e.g., [64, 128]) via policy sampling.

Each architecture is trained + validated on MNIST.

The validation accuracy becomes the reward for the controller.

Over time, the controller learns to generate better-performing architectures.
"""


import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import random
import numpy as np
import matplotlib.pyplot as plt
 
# Controller: generates architectures (layer size choices)
class NASController(nn.Module):
    def __init__(self, n_choices=3):
        super().__init__()
        self.n_layers = 2  # choose 2 hidden layers
        self.hidden_sizes = [32, 64, 128]
        self.n_choices = n_choices
        self.policy = nn.Sequential(
            nn.Linear(1, 32), nn.ReLU(),
            nn.Linear(32, self.n_layers * self.n_choices)
        )
 
    def forward(self):
        dummy_input = torch.FloatTensor([[1.0]])
        logits = self.policy(dummy_input).view(self.n_layers, self.n_choices)
        probs = torch.softmax(logits, dim=1)
        dists = [torch.distributions.Categorical(p) for p in probs]
        actions = [d.sample() for d in dists]
        log_probs = [d.log_prob(a) for d, a in zip(dists, actions)]
        arch = [self.hidden_sizes[a.item()] for a in actions]
        return arch, torch.stack(log_probs).sum()
 
# Architecture training and evaluation
def train_and_evaluate(arch, train_loader, val_loader):
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(28*28, arch[0]), nn.ReLU(),
        nn.Linear(arch[0], arch[1]), nn.ReLU(),
        nn.Linear(arch[1], 10)
    )
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()
 
    for _ in range(1):  # train for 1 epoch for speed
        for x, y in train_loader:
            out = model(x)
            loss = loss_fn(out, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
 
    # Validation accuracy
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in val_loader:
            preds = model(x).argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return correct / total
 
# Data loading (small subset for speed)
transform = transforms.ToTensor()
train_set = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
val_set = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=128, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=256, shuffle=False)
 
# NAS setup
controller = NASController()
controller_optim = optim.Adam(controller.parameters(), lr=1e-3)
rewards = []
 
# Training loop
for ep in range(30):
    arch, log_prob = controller()
    acc = train_and_evaluate(arch, train_loader, val_loader)
    reward = acc
    loss = -log_prob * reward  # maximize reward
 
    controller_optim.zero_grad()
    loss.backward()
    controller_optim.step()
 
    rewards.append(reward)
    print(f"Episode {ep+1}, Architecture: {arch}, Val Accuracy: {acc:.4f}")
 
# Plot reward trend
plt.plot(rewards)
plt.title("RL for Neural Architecture Search")
plt.xlabel("Episode")
plt.ylabel("Validation Accuracy")
plt.grid(True)
plt.show()