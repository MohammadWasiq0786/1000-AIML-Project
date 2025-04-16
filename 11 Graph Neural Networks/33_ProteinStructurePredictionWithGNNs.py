"""
Project 433. Protein structure prediction with GNNs
Description:
Proteins can be represented as graphs, where amino acids are nodes and bonds or spatial proximity define edges. Graph Neural Networks (GNNs) are powerful tools for predicting properties like secondary structure, folding class, or contact maps. In this project, we'll build a GNN that takes a protein residue graph and predicts protein class (structure/function type).

About:
✅ What It Does:
Loads protein graphs from the PROTEINS dataset.

Trains a GCN model to classify proteins into structural classes.

Uses mean pooling for graph-level predictions.

Evaluates prediction accuracy across protein graphs.
"""

import torch
import torch.nn.functional as F
from torch_geometric.datasets import TUDataset
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.loader import DataLoader
 
# 1. Load PROTEINS dataset (residues as nodes, graphs as proteins)
dataset = TUDataset(root='/tmp/PROTEINS', name='PROTEINS')
dataset = dataset.shuffle()
 
train_dataset = dataset[:900]
test_dataset = dataset[900:]
train_loader = DataLoader(train_dataset, batch_size=32)
test_loader = DataLoader(test_dataset, batch_size=32)
 
# 2. Define GCN model
class ProteinGCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(dataset.num_node_features, 64)
        self.conv2 = GCNConv(64, 64)
        self.lin1 = torch.nn.Linear(64, 32)
        self.lin2 = torch.nn.Linear(32, dataset.num_classes)
 
    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = global_mean_pool(x, batch)
        x = F.relu(self.lin1(x))
        return F.log_softmax(self.lin2(x), dim=1)
 
# 3. Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ProteinGCN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = torch.nn.NLLLoss()
 
# 4. Train function
def train():
    model.train()
    total_loss = 0
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index, batch.batch)
        loss = loss_fn(out, batch.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)
 
# 5. Test function
def test():
    model.eval()
    correct = 0
    for batch in test_loader:
        batch = batch.to(device)
        out = model(batch.x, batch.edge_index, batch.batch)
        pred = out.argmax(dim=1)
        correct += int((pred == batch.y).sum())
    return correct / len(test_dataset)
 
# 6. Run training
for epoch in range(1, 21):
    loss = train()
    acc = test()
    print(f"Epoch {epoch:02d}, Train Loss: {loss:.4f}, Test Accuracy: {acc:.4f}")