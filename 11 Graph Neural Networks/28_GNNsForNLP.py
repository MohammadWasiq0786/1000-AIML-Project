"""
Project 428. Graph neural networks for NLP
Description:
Graphs are natural representations for many NLP tasks — such as dependency parsing, semantic graphs, or knowledge graphs. With Graph Neural Networks, we can model the relationships between words, phrases, or entities more effectively. In this project, we’ll use a GCN to classify sentences based on their dependency parse trees.

About:
✅ What It Does:
Parses sentences into dependency trees using spaCy.

Constructs graphs where words are nodes, and edges are syntactic dependencies.

Applies GCN to classify sentence sentiment.

Uses mean pooling over word representations for graph-level prediction.
"""

# pip install spacy torch torch-geometric
# python -m spacy download en_core_web_sm

import torch
import torch.nn.functional as F
import spacy
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
 
# 1. Prepare NLP model
nlp = spacy.load("en_core_web_sm")
 
# 2. Create dependency graph from sentence
def sentence_to_graph(sentence):
    doc = nlp(sentence)
    edge_index = []
    x = []
 
    for token in doc:
        x.append(token.vector if token.has_vector else torch.rand(300))  # use random if no vector
        if token.i != token.head.i:  # skip root
            edge_index.append([token.i, token.head.i])
            edge_index.append([token.head.i, token.i])  # bidirectional
 
    x = torch.tensor(x, dtype=torch.float)
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    return Data(x=x, edge_index=edge_index)
 
# 3. Define GCN model for sentence graph
class SentenceGCN(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.classifier = torch.nn.Linear(hidden_dim, 2)
 
    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = x.mean(dim=0)  # mean pooling over sentence
        return F.log_softmax(self.classifier(x), dim=0)
 
# 4. Sample data
sentences = [
    ("I love this movie!", 1),
    ("This film is terrible.", 0),
    ("What an amazing experience.", 1),
    ("I wouldn't recommend it.", 0)
]
 
# 5. Training setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SentenceGCN(in_dim=300, hidden_dim=64).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = torch.nn.NLLLoss()
 
# 6. Training loop
for epoch in range(1, 51):
    model.train()
    total_loss = 0
    for text, label in sentences:
        data = sentence_to_graph(text).to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = loss_fn(out.unsqueeze(0), torch.tensor([label], dtype=torch.long, device=device))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    if epoch % 10 == 0:
        print(f"Epoch {epoch:02d}, Loss: {total_loss:.4f}")