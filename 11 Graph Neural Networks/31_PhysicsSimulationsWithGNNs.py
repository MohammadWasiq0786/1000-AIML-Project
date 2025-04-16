"""
Project 431. Physics simulations with GNNs
Description:
Graph Neural Networks can simulate physical systems by modeling particles or objects as nodes and their interactions (e.g., forces, collisions) as edges. These models learn to approximate physical dynamics, making them useful for simulations in fluid dynamics, molecular systems, or rigid-body interactions.

In this project, we’ll build a simplified particle interaction simulation where a GNN learns to predict the next velocity of each particle given positions and neighbors.

About:
✅ What It Does:
Creates a particle system where nodes attract each other.

Builds a GCN to learn velocity predictions from particle positions.

Trains the GNN to match a simple physics rule.

Visualizes the learned velocity vector field.
"""

import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import matplotlib.pyplot as plt
 
# 1. Simulate particles in 2D space
def generate_particles(num_particles=10):
    pos = torch.rand(num_particles, 2) * 10
    vel = torch.zeros_like(pos)
    edge_index = torch.combinations(torch.arange(num_particles), r=2).t()
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    return pos, vel, edge_index
 
# 2. Define GNN model to predict velocity updates
class PhysicsGNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(4, 64)
        self.conv2 = GCNConv(64, 2)  # output: delta velocity (2D)
 
    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        return self.conv2(x, edge_index)
 
# 3. Training setup
model = PhysicsGNN()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
 
# 4. Simulate target behavior: move toward average neighbor position
def compute_target_velocity(pos, edge_index):
    num_nodes = pos.size(0)
    target_vel = torch.zeros_like(pos)
    for i in range(num_nodes):
        neighbors = edge_index[1][edge_index[0] == i]
        if len(neighbors) > 0:
            avg_pos = pos[neighbors].mean(0)
            target_vel[i] = avg_pos - pos[i]
    return target_vel
 
# 5. Train GNN to mimic the physical rule
for epoch in range(1, 101):
    pos, vel, edge_index = generate_particles()
    input_features = torch.cat([pos, vel], dim=1)
    data = Data(x=input_features, edge_index=edge_index)
 
    target = compute_target_velocity(pos, edge_index)
    optimizer.zero_grad()
    pred = model(data.x, data.edge_index)
    loss = F.mse_loss(pred, target)
    loss.backward()
    optimizer.step()
 
    if epoch % 10 == 0:
        print(f"Epoch {epoch:03d}, Loss: {loss.item():.4f}")
 
# 6. Visualize particle and velocity field
with torch.no_grad():
    pred_vel = model(data.x, data.edge_index)
 
plt.figure(figsize=(6, 6))
plt.quiver(pos[:, 0], pos[:, 1], pred_vel[:, 0], pred_vel[:, 1], angles='xy', scale_units='xy', scale=1, color='blue')
plt.scatter(pos[:, 0], pos[:, 1], color='red')
plt.title("Predicted Particle Velocity Field (GNN)")
plt.xlabel("X")
plt.ylabel("Y")
plt.grid(True)
plt.show()