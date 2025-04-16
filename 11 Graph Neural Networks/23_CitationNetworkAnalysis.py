"""
Project 423. Citation network analysis
Description:
A citation network is a directed graph where nodes represent papers and edges indicate citations (i.e., one paper cites another). Analyzing such networks can reveal influential research, emerging fields, or collaboration structures. In this project, we’ll use NetworkX to analyze a synthetic citation graph and apply key metrics like in-degree, PageRank, and connected components.

About:
✅ What It Does:
Builds a directed graph simulating paper citations.

Computes in-degree as citation count.

Applies PageRank to find the most influential papers.

Identifies strongly connected components (cycles of mutual citation).

Visualizes the citation network layout.
"""


# pip install networkx matplotlib

import networkx as nx
import matplotlib.pyplot as plt
 
# 1. Create a synthetic citation graph
G = nx.DiGraph()
edges = [
    (1, 3), (2, 3), (3, 4), (4, 5), (2, 5), (1, 6), (6, 5),
    (7, 5), (8, 7), (9, 7), (10, 9), (10, 8), (3, 9)
]
G.add_edges_from(edges)
 
# 2. Analyze in-degree (citations received)
in_degrees = dict(G.in_degree())
top_cited = sorted(in_degrees.items(), key=lambda x: x[1], reverse=True)
print("Top cited papers (by in-degree):")
for paper, citations in top_cited[:5]:
    print(f"Paper {paper} → {citations} citations")
 
# 3. PageRank (importance based on recursive citation)
pagerank_scores = nx.pagerank(G, alpha=0.85)
top_papers = sorted(pagerank_scores.items(), key=lambda x: x[1], reverse=True)
print("\nMost influential papers (by PageRank):")
for paper, score in top_papers[:5]:
    print(f"Paper {paper} → PageRank: {score:.4f}")
 
# 4. Strongly connected components (mutual citation loops)
scc = list(nx.strongly_connected_components(G))
print(f"\nNumber of strongly connected components: {len(scc)}")
print("Largest SCC:", max(scc, key=len))
 
# 5. Visualize
plt.figure(figsize=(8, 6))
pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True, node_color='lightblue', arrows=True)
plt.title("Synthetic Citation Network")
plt.show()