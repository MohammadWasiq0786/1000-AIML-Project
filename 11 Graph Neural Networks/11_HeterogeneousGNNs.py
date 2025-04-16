"""
Project 411. Heterogeneous graph neural networks
Description:
Heterogeneous Graph Neural Networks handle graphs with multiple types of nodes and edges, such as knowledge graphs, academic networks (authors, papers, conferences), or e-commerce graphs (users, products, transactions). These models treat node/edge types differently and learn type-specific representations. In this project, we’ll implement a HetGNN on a heterogeneous academic graph using PyTorch Geometric.

About:
✅ What It Does:
Loads the AIFB knowledge graph with multiple node and edge types.

Uses an R-GCN to aggregate messages across different relation types.

Learns to classify entities (e.g., people or organizations) using type-aware message passing.
"""

import torch
import torch.nn.functional as F
from torch_geometric.datasets import AIFBDataset
from torch_geometric.nn import RGCNConv
from torch_geometric.loader import NeighborLoader
 
# 1. Load AIFB heterogeneous dataset (academic knowledge graph)
dataset = AIFBDataset(root='/tmp/AIFB')
data = dataset[0]
 
# 2. Encode target labels as integers
from sklearn.preprocessing import LabelEncoder
data.y = LabelEncoder().fit_transform(data.y)
 
# 3. Define R-GCN model
class RGCN(torch.nn.Module):
    def __init__(self, in_channels, out_channels, num_relations):
        super().__init__()
        self.conv1 = RGCNConv(in_channels, 16, num_relations)
        self.conv2 = RGCNConv(16, out_channels, num_relations)
 
    def forward(self, x, edge_index, edge_type):
        x = F.relu(self.conv1(x, edge_index, edge_type))
        x = self.conv2(x, edge_index, edge_type)
        return F.log_softmax(x, dim=1)
 
# 4. Set up training
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = RGCN(data.num_features, dataset.num_classes, num_relations=len(data.edge_type.unique())).to(device)
data = data.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
 
# 5. Train function
def train():
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index, data.edge_type)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()
 
# 6. Evaluation
def test():
    model.eval()
    out = model(data.x, data.edge_index, data.edge_type)
    pred = out.argmax(dim=1)
    correct = pred[data.test_mask] == data.y[data.test_mask]
    acc = int(correct.sum()) / int(data.test_mask.sum())
    return acc
 
# 7. Training loop
for epoch in range(1, 51):
    loss = train()
    acc = test()
    if epoch % 10 == 0:
        print(f"Epoch {epoch:02d}, Loss: {loss:.4f}, Test Accuracy: {acc:.4f}")