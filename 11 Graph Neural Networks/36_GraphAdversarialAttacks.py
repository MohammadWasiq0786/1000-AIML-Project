"""
Project 436. Graph adversarial attacks
Description:
Graph adversarial attacks aim to fool a trained GNN by making small perturbations to the graph — like adding/removing edges or changing node features — without altering the true label. These attacks test the robustness of GNNs and are critical for security-sensitive applications (e.g., fraud detection, recommendation systems). In this project, we’ll perform a targeted edge-perturbation attack on a trained GCN.

About:
✅ What It Does:
Trains a GCN on Cora.

Chooses a target node and stores its original prediction.

Attempts to flip one edge (add/remove connection) at a time.

If any flip causes the prediction to change → adversarial success.

Demonstrates vulnerability of GNNs to minor structural attacks.
"""

import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv
import copy
 
# 1. Load dataset
dataset = Planetoid(root='/tmp/Cora', name='Cora')
data = dataset[0]
 
# 2. Define simple GCN
class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
 
    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        return self.conv2(x, edge_index)
 
# 3. Train clean model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GCN(dataset.num_node_features, 16, dataset.num_classes).to(device)
data = data.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = torch.nn.CrossEntropyLoss()
 
def train():
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = loss_fn(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
 
for epoch in range(1, 101):
    train()
 
# 4. Predict a node before the attack
model.eval()
target_node = 10
original_pred = model(data.x, data.edge_index)[target_node].argmax().item()
print(f"Original prediction for node {target_node}: {original_pred}")
 
# 5. Adversarial attack: flip an edge (add or remove)
def flip_edge(edge_index, u, v):
    edge_list = edge_index.t().tolist()
    if [u, v] in edge_list:
        edge_list.remove([u, v])
        edge_list.remove([v, u])
    else:
        edge_list.append([u, v])
        edge_list.append([v, u])
    return torch.tensor(edge_list, dtype=torch.long).t().contiguous()
 
# Try flipping connections from node 10 to its 10 nearest neighbors
attacked_edge_index = data.edge_index
for other_node in range(len(data.x)):
    if other_node == target_node:
        continue
    test_edge_index = flip_edge(data.edge_index, target_node, other_node)
    test_model = copy.deepcopy(model)
    with torch.no_grad():
        output = test_model(data.x, test_edge_index)[target_node]
        new_pred = output.argmax().item()
        if new_pred != original_pred:
            print(f"Adversarial success: changed prediction by flipping edge ({target_node}, {other_node})")
            break