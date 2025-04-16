"""
Project 258. Imitation learning implementation
Description:
Imitation Learning is a technique where an agent learns to act by mimicking expert behavior, without explicitly learning a reward function (unlike IRL). The most basic form is Behavior Cloning, where we treat expert demonstrations as supervised learning data — training a model to map states to actions directly.

In this project, we’ll implement Behavior Cloning in a GridWorld, where an expert follows the shortest path, and the agent learns to do the same via supervised learning.

About:
✅ What It Does:
Collects expert state-action pairs from a simple policy.

Trains a neural network to predict actions via supervised learning.

Evaluates whether the learned policy can imitate the expert path.

Foundation for more advanced methods like DAgger or GAIL.
"""


import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
 
# Simple GridWorld (5x5)
class GridWorld:
    def __init__(self, size=5, goal=(4, 4)):
        self.size = size
        self.goal = goal
 
    def get_state_vector(self, pos):
        # One-hot encoding of grid cell
        vec = np.zeros(self.size * self.size)
        idx = pos[0] * self.size + pos[1]
        vec[idx] = 1
        return vec
 
# Expert policy: shortest path (Right → Down)
def generate_expert_data(start, goal, env):
    X, y = [], []
    pos = start
    while pos != goal:
        r, c = pos
        gr, gc = goal
        if c < gc:
            next_pos = (r, c + 1)
            action = 3  # Right
        else:
            next_pos = (r + 1, c)
            action = 1  # Down
        X.append(env.get_state_vector(pos))
        y.append(action)
        pos = next_pos
    return np.array(X), np.array(y)
 
# Neural network policy: input is state, output is action
class PolicyNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(),
            nn.Linear(64, output_dim)
        )
 
    def forward(self, x):
        return self.fc(x)
 
# Setup
env = GridWorld()
start, goal = (0, 0), env.goal
X_train, y_train = generate_expert_data(start, goal, env)
 
# Convert to PyTorch tensors
X_tensor = torch.FloatTensor(X_train)
y_tensor = torch.LongTensor(y_train)
 
# Initialize model
model = PolicyNet(input_dim=env.size * env.size, output_dim=4)  # 4 actions: U,D,L,R
optimizer = optim.Adam(model.parameters(), lr=1e-2)
loss_fn = nn.CrossEntropyLoss()
 
# Train behavior cloning model
losses = []
for epoch in range(200):
    logits = model(X_tensor)
    loss = loss_fn(logits, y_tensor)
 
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    losses.append(loss.item())
 
print("✅ Behavior cloning policy trained from expert.")
 
# Test learned policy
def test_policy(env, model, start, goal):
    state = start
    path = [state]
    for _ in range(20):
        vec = torch.FloatTensor(env.get_state_vector(state)).unsqueeze(0)
        with torch.no_grad():
            action = torch.argmax(model(vec)).item()
 
        # Action: 0=Up, 1=Down, 2=Left, 3=Right
        r, c = state
        if action == 0: r = max(0, r - 1)
        elif action == 1: r = min(env.size - 1, r + 1)
        elif action == 2: c = max(0, c - 1)
        elif action == 3: c = min(env.size - 1, c + 1)
 
        state = (r, c)
        path.append(state)
        if state == goal:
            break
    return path
 
# Run and visualize
learned_path = test_policy(env, model, start, goal)
grid = np.zeros((env.size, env.size))
for r, c in learned_path:
    grid[r, c] = 1
 
plt.imshow(grid, cmap='Blues')
plt.title("Behavior Cloning: Learned Path to Goal")
plt.xticks(range(env.size))
plt.yticks(range(env.size))
plt.grid(True)
plt.show()