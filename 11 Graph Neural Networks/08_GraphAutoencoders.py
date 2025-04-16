"""
Project 408. Graph autoencoders
Description:
Graph Autoencoders (GAEs) learn low-dimensional embeddings of graph nodes in an unsupervised way by trying to reconstruct the graph structure (usually its edges). These embeddings can be used for node clustering, link prediction, or visualization. In this project, we'll implement a GAE using PyTorch Geometric on the Cora dataset.

About:
✅ What It Does:
Builds a graph autoencoder that compresses nodes into embeddings by minimizing reconstruction loss on the graph structure.

The model learns to predict which edges exist by reconstructing a binary adjacency matrix.

Can be used for link prediction and unsupervised node embedding.
"""

import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv, GAE
from torch_geometric.utils import train_test_split_edges
 
# 1. Load and prepare the Cora dataset
dataset = Planetoid(root='/tmp/Cora', name='Cora')
data = dataset[0]
data = train_test_split_edges(data)  # Split edges for unsupervised training
 
# 2. Define a GCN-based encoder for the GAE
class Encoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels)
        self.conv2 = GCNConv(2 * out_channels, out_channels)
 
    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        return self.conv2(x, edge_index)
 
# 3. Initialize model and optimizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GAE(Encoder(dataset.num_node_features, 64)).to(device)
x, train_edge_index = data.x.to(device), data.train_pos_edge_index.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
 
# 4. Training function
def train():
    model.train()
    optimizer.zero_grad()
    z = model.encode(x, train_edge_index)
    loss = model.recon_loss(z, train_edge_index)  # Binary cross-entropy on edges
    loss.backward()
    optimizer.step()
    return loss.item()
 
# 5. Evaluation (e.g., for link prediction)
@torch.no_grad()
def test():
    model.eval()
    z = model.encode(x, train_edge_index)
    auc = model.test(z, data.test_pos_edge_index, data.test_neg_edge_index)
    return auc
 
# 6. Train loop
for epoch in range(1, 101):
    loss = train()
    if epoch % 10 == 0:
        auc = test()
        print(f"Epoch {epoch:03d}, Loss: {loss:.4f}, Test AUC: {auc:.4f}")