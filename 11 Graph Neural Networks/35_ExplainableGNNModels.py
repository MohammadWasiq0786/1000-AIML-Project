"""
Project 435. Explainable GNN models
Description:
Explainability in Graph Neural Networks is critical for high-stakes applications like finance, healthcare, or law. Understanding why a GNN made a certain prediction helps improve trust and transparency. In this project, we’ll use GNNExplainer, a method that identifies the most relevant subgraph and node features contributing to a prediction.

We’ll demonstrate this using a node classification task on the Cora citation dataset.

About:
✅ What It Does:
Trains a GCN on the Cora citation graph.

Uses GNNExplainer to identify key nodes and edges influencing a prediction.

Visualizes the important subgraph contributing to a classification decision.

Demonstrates how GNN predictions can be made interpretable.
"""

import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv, GNNExplainer
import matplotlib.pyplot as plt
 
# 1. Load dataset
dataset = Planetoid(root='/tmp/Cora', name='Cora')
data = dataset[0]
 
# 2. Define a simple GCN model
class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
 
    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        return self.conv2(x, edge_index)
 
# 3. Model setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GCN(dataset.num_node_features, 16, dataset.num_classes).to(device)
data = data.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = torch.nn.CrossEntropyLoss()
 
# 4. Training loop
def train():
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = loss_fn(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
 
# Train the model
for epoch in range(1, 101):
    train()
 
# 5. Explain prediction for a single node
explainer = GNNExplainer(model, epochs=200)
node_idx = 10  # pick a node to explain
model.eval()
node_feat_mask, edge_mask = explainer.explain_node(node_idx, data.x, data.edge_index)
 
# 6. Visualize explanation subgraph
ax, G = explainer.visualize_subgraph(node_idx, data.edge_index, edge_mask, y=data.y)
plt.title(f"Explanation for node {node_idx}")
plt.show()