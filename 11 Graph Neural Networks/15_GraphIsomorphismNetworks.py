"""
Project 415. Graph isomorphism networks
Description:
Graph Isomorphism Networks (GINs) are powerful GNNs that match the discriminative power of the Weisfeiler-Lehman graph isomorphism test. Unlike traditional GCNs that average neighbor features, GINs sum them and pass through an MLP, making them ideal for graph classification tasks. In this project, we’ll implement a GIN on the MUTAG dataset using PyTorch Geometric.

About:
✅ What It Does:
Implements a 2-layer GIN model with MLPs inside each convolution.

Uses global sum pooling to generate graph-level embeddings.

Trains the model on MUTAG, a molecular graph classification dataset.
"""

# pip install torch-geometric

import torch
import torch.nn.functional as F
from torch_geometric.datasets import TUDataset
from torch_geometric.nn import GINConv, global_add_pool
from torch_geometric.loader import DataLoader
 
# 1. Load MUTAG dataset
dataset = TUDataset(root='/tmp/MUTAG', name='MUTAG')
loader = DataLoader(dataset, batch_size=32, shuffle=True)
 
# 2. Define the GIN model
class GIN(torch.nn.Module):
    def __init__(self, num_features, hidden_dim, num_classes):
        super().__init__()
        nn1 = torch.nn.Sequential(torch.nn.Linear(num_features, hidden_dim), torch.nn.ReLU(), torch.nn.Linear(hidden_dim, hidden_dim))
        self.conv1 = GINConv(nn1)
        nn2 = torch.nn.Sequential(torch.nn.Linear(hidden_dim, hidden_dim), torch.nn.ReLU(), torch.nn.Linear(hidden_dim, hidden_dim))
        self.conv2 = GINConv(nn2)
        self.lin1 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.lin2 = torch.nn.Linear(hidden_dim, num_classes)
 
    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = global_add_pool(x, batch)  # Sum pooling
        x = F.relu(self.lin1(x))
        return F.log_softmax(self.lin2(x), dim=1)
 
# 3. Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GIN(dataset.num_node_features, 64, dataset.num_classes).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = torch.nn.NLLLoss()
 
# 4. Training function
def train():
    model.train()
    total_loss = 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index, batch.batch)
        loss = loss_fn(out, batch.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)
 
# 5. Training loop
for epoch in range(1, 31):
    loss = train()
    print(f"Epoch {epoch:02d}, Loss: {loss:.4f}")