"""
Project 419. Graph few-shot learning
Description:
Graph Few-Shot Learning tackles the problem of learning from very few labeled examples per class on graph-structured data. It’s especially valuable in domains like bioinformatics, social networks, or fraud detection, where labeling is expensive. A common approach uses meta-learning (e.g., Prototypical Networks) combined with GNNs to generalize quickly to new tasks.

In this project, we’ll simulate few-shot learning using a GCN encoder + Prototypical Network on a synthetic citation-like graph dataset.

About:
✅ What It Does:
Uses a GCN encoder to embed graph nodes.

Simulates a 5-way 1-shot few-shot learning task from the Cora dataset.

Applies a prototypical loss to predict query nodes based on a few support examples.

Trains the model to generalize quickly across few-shot graph tasks.
"""


# pip install torch-geometric

import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv
import random
 
# 1. Load Cora dataset
dataset = Planetoid(root='/tmp/Cora', name='Cora')
data = dataset[0]
 
# 2. Define GCN encoder
class GCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
 
    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        return self.conv2(x, edge_index)
 
# 3. Prototypical loss for few-shot learning
def prototypical_loss(embeddings, labels, support_idx, query_idx):
    support_emb = embeddings[support_idx]
    support_labels = labels[support_idx]
    query_emb = embeddings[query_idx]
    query_labels = labels[query_idx]
 
    classes = support_labels.unique()
    prototypes = torch.stack([
        support_emb[support_labels == c].mean(0) for c in classes
    ])
 
    dists = torch.cdist(query_emb, prototypes)
    preds = dists.argmin(dim=1)
    acc = (classes[preds] == query_labels).float().mean().item()
    loss = F.cross_entropy(-dists, torch.tensor([classes.tolist().index(y.item()) for y in query_labels]))
    return loss, acc
 
# 4. Sample few-shot task (5-way 1-shot)
def sample_few_shot(labels, num_classes=5, k_shot=1, q_query=5):
    classes = torch.unique(labels)
    selected = random.sample(classes.tolist(), num_classes)
    support_idx = []
    query_idx = []
 
    for c in selected:
        idx = (labels == c).nonzero(as_tuple=True)[0].tolist()
        chosen = random.sample(idx, k_shot + q_query)
        support_idx.extend(chosen[:k_shot])
        query_idx.extend(chosen[k_shot:])
 
    return torch.tensor(support_idx), torch.tensor(query_idx)
 
# 5. Training loop
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GCNEncoder(dataset.num_node_features, 64).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
data = data.to(device)
 
for epoch in range(1, 101):
    model.train()
    optimizer.zero_grad()
    emb = model(data.x, data.edge_index)
    support_idx, query_idx = sample_few_shot(data.y.cpu(), num_classes=5, k_shot=1, q_query=5)
    loss, acc = prototypical_loss(emb, data.y, support_idx.to(device), query_idx.to(device))
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0:
        print(f"Epoch {epoch:03d}, Loss: {loss:.4f}, Few-Shot Accuracy: {acc:.4f}")