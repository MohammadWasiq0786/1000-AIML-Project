"""
Project 409. Graph generative models
Description:
Graph generative models aim to learn the underlying distribution of graphs and generate new graphs (or nodes/edges) that resemble real ones. These are useful in tasks like molecule generation, social graph synthesis, and network simulations. In this project, we’ll build a simple variational graph autoencoder (VGAE) using PyTorch Geometric to generate graphs by learning their latent structure.

About:
✅ What It Does:
Implements a Variational Graph Autoencoder (VGAE) to learn probabilistic embeddings.

Uses a GCN encoder to learn both mean and log-variance vectors.

Optimizes the reconstruction loss and KL divergence to generate new edges.

Suitable for graph generation, link prediction, and sampling new graphs.
"""

import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv, VGAE
from torch_geometric.utils import train_test_split_edges
 
# 1. Load dataset and prepare it
dataset = Planetoid(root='/tmp/Cora', name='Cora')
data = dataset[0]
data = train_test_split_edges(data)  # Prepares train/test edges
 
# 2. Define encoder for VGAE
class VariationalEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels)
        self.conv_mu = GCNConv(2 * out_channels, out_channels)
        self.conv_logvar = GCNConv(2 * out_channels, out_channels)
 
    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        return self.conv_mu(x, edge_index), self.conv_logvar(x, edge_index)
 
# 3. Initialize VGAE model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = VGAE(VariationalEncoder(dataset.num_node_features, 64)).to(device)
x = data.x.to(device)
train_edge_index = data.train_pos_edge_index.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
 
# 4. Train function
def train():
    model.train()
    optimizer.zero_grad()
    z = model.encode(x, train_edge_index)
    loss = model.recon_loss(z, train_edge_index)
    loss = loss + (1 / data.num_nodes) * model.kl_loss()
    loss.backward()
    optimizer.step()
    return loss.item()
 
# 5. Test function
@torch.no_grad()
def test():
    model.eval()
    z = model.encode(x, train_edge_index)
    auc = model.test(z, data.test_pos_edge_index, data.test_neg_edge_index)
    return auc
 
# 6. Training loop
for epoch in range(1, 101):
    loss = train()
    if epoch % 10 == 0:
        auc = test()
        print(f"Epoch {epoch:03d}, Loss: {loss:.4f}, AUC: {auc:.4f}")