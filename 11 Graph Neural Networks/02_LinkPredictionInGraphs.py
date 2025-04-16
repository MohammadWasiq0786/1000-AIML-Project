"""
Project 402. Link prediction in graphs
Description:
Link prediction aims to predict missing or future edges between nodes in a graph. It's useful in recommendation systems, social networks, and biological networks — for instance, suggesting new friends or predicting protein interactions. In this project, we’ll build a model to determine whether an edge exists between two nodes using graph embeddings.

About:
✅ What It Does:
Splits edges into train/test sets for supervised learning.

Learns node embeddings using a GCN encoder.

Uses dot product as a similarity score for predicting if an edge exists between two nodes.

Evaluates the model using ROC-AUC, a common metric for binary classification.
"""

import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv
from torch_geometric.utils import train_test_split_edges, negative_sampling, remove_self_loops, add_self_loops
 
# 1. Load dataset and prepare it for link prediction
dataset = Planetoid(root='/tmp/Cora', name='Cora')
data = dataset[0]
data = train_test_split_edges(data)  # Creates data.train_pos_edge_index, etc.
 
# 2. Define GCN model (encoder)
class GCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCNEncoder, self).__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels)
        self.conv2 = GCNConv(2 * out_channels, out_channels)
 
    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        return self.conv2(x, edge_index)
 
# 3. Dot-product decoder
def decode(z, edge_index):
    # z is node embedding; edge_index defines pairs
    return (z[edge_index[0]] * z[edge_index[1]]).sum(dim=1)
 
# 4. Link prediction model setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GCNEncoder(dataset.num_node_features, 64).to(device)
x, train_pos_edge_index = data.x.to(device), data.train_pos_edge_index.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
 
# 5. Training step
def train():
    model.train()
    optimizer.zero_grad()
    z = model(x, train_pos_edge_index)
 
    pos_score = decode(z, data.train_pos_edge_index.to(device))
    pos_loss = -F.logsigmoid(pos_score).mean()
 
    # Sample negative edges
    neg_edge_index = negative_sampling(edge_index=train_pos_edge_index, num_nodes=x.size(0))
    neg_score = decode(z, neg_edge_index.to(device))
    neg_loss = -F.logsigmoid(-neg_score).mean()
 
    loss = pos_loss + neg_loss
    loss.backward()
    optimizer.step()
    return loss.item()
 
# 6. Evaluation using AUC
from sklearn.metrics import roc_auc_score
 
@torch.no_grad()
def test():
    model.eval()
    z = model(x, train_pos_edge_index)
 
    # Positive edges
    pos_score = decode(z, data.test_pos_edge_index.to(device)).cpu()
    pos_labels = torch.ones(pos_score.size(0))
 
    # Negative edges
    neg_score = decode(z, data.test_neg_edge_index.to(device)).cpu()
    neg_labels = torch.zeros(neg_score.size(0))
 
    scores = torch.cat([pos_score, neg_score])
    labels = torch.cat([pos_labels, neg_labels])
 
    return roc_auc_score(labels, scores)
 
# 7. Run training loop
for epoch in range(1, 101):
    loss = train()
    if epoch % 10 == 0:
        auc = test()
        print(f"Epoch {epoch:03d}, Loss: {loss:.4f}, AUC: {auc:.4f}")