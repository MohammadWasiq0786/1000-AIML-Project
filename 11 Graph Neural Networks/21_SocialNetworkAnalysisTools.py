"""
Project 421. Social network analysis tools
Description:
Social Network Analysis (SNA) helps uncover patterns, influence, and structure within social systems using graph theory. With tools like degree centrality, betweenness, clustering coefficient, and community detection, we can study everything from influential users to information diffusion.

In this project, we’ll use NetworkX to explore a synthetic social network and apply core SNA metrics.

About:
✅ What It Does:
Creates a scale-free social graph using preferential attachment.

Computes key SNA metrics: degree, betweenness, closeness, clustering.

Prints and visualizes influential nodes based on centrality.

Helps interpret the structure and hierarchy in a social network.
"""


# pip install networkx matplotlib

import networkx as nx
import matplotlib.pyplot as plt
 
# 1. Generate a scale-free social network (Barabási–Albert model)
G = nx.barabasi_albert_graph(n=50, m=2)  # 50 nodes, 2 new edges per new node
 
# 2. Calculate centrality measures
degree_centrality = nx.degree_centrality(G)
betweenness_centrality = nx.betweenness_centrality(G)
closeness_centrality = nx.closeness_centrality(G)
clustering_coeff = nx.clustering(G)
 
# 3. Print top influencers by degree centrality
top_influencers = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:5]
print("Top influencers (by degree centrality):")
for node, score in top_influencers:
    print(f"Node {node}: {score:.3f}")
 
# 4. Visualize network with node size based on centrality
node_sizes = [1000 * degree_centrality[n] for n in G.nodes()]
nx.draw(G, with_labels=True, node_color='skyblue', node_size=node_sizes, edge_color='gray')
plt.title("Social Network with Node Size ~ Degree Centrality")
plt.show()
 
# 5. Network stats
print(f"Number of nodes: {G.number_of_nodes()}")
print(f"Number of edges: {G.number_of_edges()}")
print(f"Average clustering coefficient: {nx.average_clustering(G):.3f}")