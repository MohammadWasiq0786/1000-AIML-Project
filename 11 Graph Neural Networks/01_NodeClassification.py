"""
Project 401. Node classification implementation
Description:
Node classification is a fundamental task in graph learning where the goal is to predict the label of a node based on its features and graph structure. It's widely used in applications like social network analysis, citation networks, and biological graphs. In this project, we'll use a Graph Convolutional Network (GCN) to perform node classification on a well-known graph dataset — Cora.

About:
✅ What It Does:
Loads the Cora citation graph, where nodes represent papers and edges represent citations.

Builds a 2-layer Graph Convolutional Network to learn node embeddings.

Uses those embeddings to predict node labels.

Evaluates performance on train, validation, and test splits.
"""

import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv
 
# 1. Load the Cora dataset (a citation network graph)
dataset = Planetoid(root='/tmp/Cora', name='Cora')
data = dataset[0]  # The graph data
 
# 2. Define the GCN model
class GCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)
 
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)
 
# 3. Initialize the model, optimizer, and loss
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GCN(dataset.num_node_features, 16, dataset.num_classes).to(device)
data = data.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
 
# 4. Training loop
def train():
    model.train()
    optimizer.zero_grad()
    out = model(data)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()
 
# 5. Testing function
def test():
    model.eval()
    logits = model(data)
    accs = []
    for mask in [data.train_mask, data.val_mask, data.test_mask]:
        pred = logits[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        accs.append(acc)
    return accs
 
# 6. Run training
for epoch in range(1, 201):
    loss = train()
    train_acc, val_acc, test_acc = test()
    if epoch % 20 == 0:
        print(f"Epoch {epoch:03d}, Loss: {loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, Test Acc: {test_acc:.4f}")