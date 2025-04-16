"""
Project 427. Anomaly detection in graphs
Description:
Anomaly detection in graphs focuses on identifying nodes, edges, or subgraphs that deviate from expected patterns — such as fake users in social networks, faulty sensors in IoT graphs, or suspicious accounts in financial networks. In this project, we'll use Graph Autoencoders (GAE) to detect structural anomalies by measuring how poorly a node’s neighborhood can be reconstructed.

About:
✅ What It Does:
Builds a Graph Autoencoder (GAE) to reconstruct the adjacency matrix.

Measures each node’s reconstruction error as an anomaly score.

Detects nodes with unusual neighborhoods based on structural mismatch.

Identifies top anomalous nodes for further investigation.
"""

import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv, GAE
from torch_geometric.utils import train_test_split_edges
 
# 1. Load Cora dataset and prepare for link prediction
dataset = Planetoid(root='/tmp/Cora', name='Cora')
data = dataset[0]
data = train_test_split_edges(data)
 
# 2. Define GCN encoder for GAE
class Encoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels)
        self.conv2 = GCNConv(2 * out_channels, out_channels)
 
    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        return self.conv2(x, edge_index)
 
# 3. Setup model and training
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GAE(Encoder(dataset.num_node_features, 64)).to(device)
x = data.x.to(device)
train_edge_index = data.train_pos_edge_index.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
 
# 4. Train function
def train():
    model.train()
    optimizer.zero_grad()
    z = model.encode(x, train_edge_index)
    loss = model.recon_loss(z, train_edge_index)
    loss.backward()
    optimizer.step()
    return loss.item()
 
# 5. Compute anomaly score as reconstruction error
@torch.no_grad()
def anomaly_scores():
    model.eval()
    z = model.encode(x, train_edge_index)
    recon_adj = torch.sigmoid(torch.matmul(z, z.t()))
    scores = ((recon_adj - torch.eye(len(z)).to(device)) ** 2).sum(dim=1)
    return scores.cpu()
 
# 6. Train and evaluate
for epoch in range(1, 101):
    loss = train()
    if epoch % 10 == 0:
        scores = anomaly_scores()
        top_anomalies = scores.topk(5).indices
        print(f"Epoch {epoch:03d}, Loss: {loss:.4f}, Top anomalous nodes: {top_anomalies.tolist()}")