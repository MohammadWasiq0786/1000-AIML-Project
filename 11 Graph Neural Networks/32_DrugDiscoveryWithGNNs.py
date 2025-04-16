"""
Project 432. Drug discovery with GNNs
Description:
In drug discovery, molecules are represented as graphs where atoms are nodes and chemical bonds are edges. GNNs are ideal for predicting molecular properties, binding affinities, or toxicity — tasks critical to evaluating drug candidates. In this project, we’ll use a GCN model on the MoleculeNet ESOL dataset to predict aqueous solubility of small molecules (a key drug property).

About:
✅ What It Does:
Loads molecular graphs and their solubility labels from the ESOL dataset.

Builds a GCN model to extract atom-level features.

Uses global mean pooling to get molecule-level embeddings.

Predicts solubility (regression) and reports MSE on test data.
"""

import torch
import torch.nn.functional as F
from torch_geometric.datasets import MoleculeNet
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.loader import DataLoader
 
# 1. Load the ESOL dataset for solubility prediction
dataset = MoleculeNet(root='/tmp/ESOL', name='ESOL')
dataset = dataset.shuffle()
 
# Split into train/test
train_dataset = dataset[:80]
test_dataset = dataset[80:]
train_loader = DataLoader(train_dataset, batch_size=16)
test_loader = DataLoader(test_dataset, batch_size=16)
 
# 2. Define GCN model for regression
class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(dataset.num_node_features, 64)
        self.conv2 = GCNConv(64, 64)
        self.lin1 = torch.nn.Linear(64, 32)
        self.lin2 = torch.nn.Linear(32, 1)
 
    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = global_mean_pool(x, batch)
        x = F.relu(self.lin1(x))
        return self.lin2(x).squeeze()
 
# 3. Training setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GCN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = torch.nn.MSELoss()
 
# 4. Train function
def train():
    model.train()
    total_loss = 0
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        pred = model(batch.x, batch.edge_index, batch.batch)
        loss = loss_fn(pred, batch.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)
 
# 5. Test function
def test():
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            pred = model(batch.x, batch.edge_index, batch.batch)
            loss = loss_fn(pred, batch.y)
            total_loss += loss.item()
    return total_loss / len(test_loader)
 
# 6. Run training loop
for epoch in range(1, 21):
    train_loss = train()
    test_loss = test()
    print(f"Epoch {epoch:02d}, Train MSE: {train_loss:.4f}, Test MSE: {test_loss:.4f}")