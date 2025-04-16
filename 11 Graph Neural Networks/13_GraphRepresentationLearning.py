"""
Project 413. Graph representation learning
Description:
Graph Representation Learning aims to learn meaningful low-dimensional vector embeddings of nodes, edges, or entire graphs while preserving structural and semantic information. These embeddings enable downstream tasks like node classification, link prediction, and graph classification. In this project, we’ll implement unsupervised graph representation learning using DeepWalk, which combines random walks with Skip-gram embeddings (like Word2Vec).

About:
✅ What It Does:
Simulates random walks on the graph to generate sequences like sentences.

Uses Word2Vec (Skip-gram) to learn embeddings by treating nodes as words.

Provides unsupervised node embeddings useful for downstream tasks.

Optionally visualizes embeddings using t-SNE for interpretability.
"""

import networkx as nx
import random
from tqdm import tqdm
from gensim.models import Word2Vec
 
# 1. Create a graph (Zachary's Karate Club)
G = nx.karate_club_graph()
 
# 2. Perform random walks (DeepWalk)
def generate_random_walks(graph, num_walks=10, walk_length=20):
    walks = []
    nodes = list(graph.nodes())
    for _ in range(num_walks):
        random.shuffle(nodes)
        for node in nodes:
            walk = [str(node)]
            while len(walk) < walk_length:
                cur = int(walk[-1])
                neighbors = list(graph.neighbors(cur))
                if not neighbors:
                    break
                next_node = random.choice(neighbors)
                walk.append(str(next_node))
            walks.append(walk)
    return walks
 
walks = generate_random_walks(G)
 
# 3. Train Word2Vec model on walks (Skip-gram)
model = Word2Vec(
    sentences=walks,
    vector_size=64,
    window=5,
    min_count=0,
    sg=1,  # Use Skip-gram
    workers=2,
    epochs=10
)
 
# 4. Get embeddings for each node
node_embeddings = {int(node): model.wv[node] for node in model.wv.index_to_key}
 
# 5. Visualize with t-SNE (optional)
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
 
X = [node_embeddings[n] for n in sorted(node_embeddings)]
X_tsne = TSNE(n_components=2).fit_transform(X)
 
plt.figure(figsize=(8, 6))
for i, coord in enumerate(X_tsne):
    plt.scatter(coord[0], coord[1])
    plt.annotate(str(i), (coord[0], coord[1]))
plt.title("DeepWalk Node Embeddings (t-SNE Projection)")
plt.show()