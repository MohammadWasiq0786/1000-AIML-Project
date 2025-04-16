"""
Project 417. Graph diffusion models
Description:
Graph Diffusion Models capture the way information, influence, or signals spread across a graph. They're essential in applications like viral marketing, epidemic modeling, and semi-supervised learning. These models often compute a diffused representation using techniques like Personalized PageRank, heat kernels, or Laplacian smoothing. In this project, we'll use Graph Diffusion Convolution (GDC) in PyTorch Geometric to improve node classification.

About:
✅ What It Does:
Applies Graph Diffusion Convolution (GDC) using Personalized PageRank.

Enhances feature smoothing across graph neighborhoods.

Improves performance on node classification tasks using a GCN.
"""

# pip install torch-geometric

import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv
from torch_geometric.transforms import GDC
 
# 1. Load Cora dataset with Graph Diffusion Convolution (GDC)
transform = GDC(self_loop_weight=1, normalization_in='sym', normalization_out='col',
                diffusion_kwargs=dict(method='ppr', alpha=0.15), sparsification_kwargs=dict(method='threshold', eps=0.01))
 
dataset = Planetoid(root='/tmp/Cora', name='Cora', transform=transform)
data = dataset[0]
 
# 2. Define a GCN model
class DiffusionGCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels, cached=True)
        self.conv2 = GCNConv(hidden_channels, out_channels, cached=True)
 
    def forward(self, data):
        x = F.relu(self.conv1(data.x, data.edge_index, data.edge_weight))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, data.edge_index, data.edge_weight)
        return F.log_softmax(x, dim=1)
 
# 3. Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = DiffusionGCN(dataset.num_node_features, 64, dataset.num_classes).to(device)
data = data.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
 
# 4. Training
def train():
    model.train()
    optimizer.zero_grad()
    out = model(data)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()
 
# 5. Testing
def test():
    model.eval()
    logits = model(data)
    accs = []
    for mask in [data.train_mask, data.val_mask, data.test_mask]:
        pred = logits[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        accs.append(acc)
    return accs
 
# 6. Run training loop
for epoch in range(1, 201):
    loss = train()
    train_acc, val_acc, test_acc = test()
    if epoch % 20 == 0:
        print(f"Epoch {epoch:03d}, Loss: {loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, Test Acc: {test_acc:.4f}")