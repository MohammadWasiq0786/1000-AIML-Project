"""
Project 407. GraphSAGE implementation
Description:
GraphSAGE (Graph Sample and AggregatE) is a scalable GNN framework that samples a fixed-size neighborhood of each node and applies aggregation functions (mean, LSTM, pooling). It enables inductive learning, making it suitable for large graphs or unseen nodes during inference. In this project, we’ll use PyTorch Geometric to implement GraphSAGE for node classification on the Cora dataset.

About:
✅ What It Does:
Implements a GraphSAGE model with two SAGEConv layers.

Learns to classify nodes in the Cora graph by aggregating neighborhood features using mean-based aggregation.

Supports inductive learning, which is useful for dynamic or unseen graphs.
"""

import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import SAGEConv
 
# 1. Load the Cora dataset
dataset = Planetoid(root='/tmp/Cora', name='Cora')
data = dataset[0]
 
# 2. Define the GraphSAGE model
class GraphSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GraphSAGE, self).__init__()
        self.sage1 = SAGEConv(in_channels, hidden_channels)
        self.sage2 = SAGEConv(hidden_channels, out_channels)
 
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.sage1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.sage2(x, edge_index)
        return F.log_softmax(x, dim=1)
 
# 3. Setup model and training environment
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GraphSAGE(dataset.num_node_features, 64, dataset.num_classes).to(device)
data = data.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
 
# 4. Training function
def train():
    model.train()
    optimizer.zero_grad()
    out = model(data)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()
 
# 5. Evaluation function
def test():
    model.eval()
    out = model(data)
    accs = []
    for mask in [data.train_mask, data.val_mask, data.test_mask]:
        pred = out[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        accs.append(acc)
    return accs
 
# 6. Train the model
for epoch in range(1, 201):
    loss = train()
    train_acc, val_acc, test_acc = test()
    if epoch % 20 == 0:
        print(f"Epoch {epoch:03d}, Loss: {loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, Test Acc: {test_acc:.4f}")