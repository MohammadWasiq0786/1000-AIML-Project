"""
Project 403. Graph clustering implementation
Description:
Graph clustering involves grouping nodes into clusters such that nodes within a cluster are more densely connected with each other than with nodes in other clusters. It's used in community detection, social analysis, and biological networks. In this project, we’ll implement unsupervised graph clustering using a Graph Autoencoder (GAE) to learn node embeddings and apply K-Means clustering.

About:
✅ What It Does:
Learns low-dimensional node embeddings using a graph autoencoder (GAE).

Applies K-Means clustering to group similar nodes together.

Evaluates clustering quality using Normalized Mutual Information (NMI).
"""

import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv, GAE
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score as NMI
 
# 1. Load Cora dataset
dataset = Planetoid(root='/tmp/Cora', name='Cora')
data = dataset[0]
 
# 2. Define GCN encoder for Graph Autoencoder
class GCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCNEncoder, self).__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels)
        self.conv2 = GCNConv(2 * out_channels, out_channels)
 
    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        return self.conv2(x, edge_index)
 
# 3. Initialize model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GAE(GCNEncoder(dataset.num_node_features, 64)).to(device)
data = data.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
 
# 4. Train autoencoder to learn node embeddings
def train():
    model.train()
    optimizer.zero_grad()
    z = model.encode(data.x, data.edge_index)
    loss = model.recon_loss(z, data.edge_index)
    loss.backward()
    optimizer.step()
    return loss.item()
 
# 5. Training loop
for epoch in range(1, 101):
    loss = train()
    if epoch % 10 == 0:
        print(f"Epoch {epoch:03d}, Loss: {loss:.4f}")
 
# 6. Perform clustering with KMeans on learned embeddings
model.eval()
z = model.encode(data.x, data.edge_index).detach().cpu().numpy()
kmeans = KMeans(n_clusters=dataset.num_classes, random_state=42)
clusters = kmeans.fit_predict(z)
 
# 7. Evaluate with NMI (Normalized Mutual Information)
true_labels = data.y.cpu().numpy()
nmi_score = NMI(true_labels, clusters)
print(f"\nClustering NMI Score: {nmi_score:.4f}")