"""
Project 412. Knowledge graph embeddings
Description:
Knowledge Graph Embeddings (KGE) represent entities and relations in a continuous vector space, preserving semantic relationships from the graph. These embeddings are essential for tasks like link prediction, question answering, and reasoning over knowledge bases. In this project, we'll implement and train a basic KGE model using the popular TransE algorithm via PyTorch Geometric.

About:
✅ What It Does:
Implements TransE: a translation-based embedding model for knowledge graphs.

Learns embeddings where head + relation ≈ tail holds true in vector space.

Samples negative triples by corrupting tails to train via margin ranking loss.
"""

# pip install torch-geometric torch-scatter torch-sparse torch-cluster

import torch
import torch.nn as nn
from torch_geometric.datasets import FB15k
from torch_geometric.transforms import ToUndirected
from torch_geometric.loader import LinkNeighborLoader
 
# 1. Load FB15k knowledge graph dataset
dataset = FB15k(root='/tmp/FB15k', transform=ToUndirected())
data = dataset[0]
 
# 2. TransE-style KGE model
class TransE(nn.Module):
    def __init__(self, num_entities, num_relations, embedding_dim=100):
        super().__init__()
        self.entity_embeddings = nn.Embedding(num_entities, embedding_dim)
        self.relation_embeddings = nn.Embedding(num_relations, embedding_dim)
 
    def forward(self, head, rel, tail):
        head_embed = self.entity_embeddings(head)
        rel_embed = self.relation_embeddings(rel)
        tail_embed = self.entity_embeddings(tail)
        score = -torch.norm(head_embed + rel_embed - tail_embed, p=1, dim=1)  # TransE score
        return score
 
# 3. Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TransE(data.num_nodes, int(data.edge_type.max()) + 1).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.MarginRankingLoss(margin=1.0)
 
# 4. Generate positive & negative triples for training
edge_index = data.edge_index
edge_type = data.edge_type
num_edges = edge_index.size(1)
 
def sample_negatives(edge_index, edge_type, num_nodes):
    corrupt_tail = edge_index.clone()
    corrupt_tail[1] = torch.randint(0, num_nodes, (num_edges,))
    return corrupt_tail, edge_type
 
# 5. Training loop
for epoch in range(1, 101):
    model.train()
    optimizer.zero_grad()
 
    pos_head = edge_index[0]
    pos_tail = edge_index[1]
    pos_rel = edge_type
 
    neg_edge_index, neg_rel = sample_negatives(edge_index, edge_type, data.num_nodes)
 
    pos_score = model(pos_head.to(device), pos_rel.to(device), pos_tail.to(device))
    neg_score = model(neg_edge_index[0].to(device), neg_rel.to(device), neg_edge_index[1].to(device))
 
    y = torch.ones(pos_score.size()).to(device)
    loss = loss_fn(pos_score, neg_score, y)
    loss.backward()
    optimizer.step()
 
    if epoch % 10 == 0:
        print(f"Epoch {epoch:03d}, Loss: {loss.item():.4f}")