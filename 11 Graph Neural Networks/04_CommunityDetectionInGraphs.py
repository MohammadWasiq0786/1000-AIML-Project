"""
Project 404. Community detection in graphs
Description:
Community detection aims to find groups of nodes (communities) in a graph that are more densely connected internally than externally. It is used in social networks, citation networks, biological systems, and influence mapping. In this project, we'll use a modularity-based approach and Louvain algorithm, which is widely used for efficient community detection.

About:
✅ What It Does:
Uses Louvain algorithm to detect communities based on modularity maximization.

Applies it on the Zachary’s Karate Club graph (a classic dataset for community detection).

Visualizes communities with distinct colors.
"""

import networkx as nx
import matplotlib.pyplot as plt
import community as community_louvain  # This is the python-louvain package
 
# 1. Create a sample graph or load one
G = nx.karate_club_graph()  # A well-known social graph with community structure
 
# 2. Apply Louvain community detection
partition = community_louvain.best_partition(G)
 
# 3. Visualize the communities
pos = nx.spring_layout(G)
plt.figure(figsize=(8, 6))
 
# Color nodes by community
for community_id in set(partition.values()):
    members = [node for node in partition if partition[node] == community_id]
    nx.draw_networkx_nodes(G, pos, nodelist=members, label=f"Community {community_id}", node_size=300)
 
nx.draw_networkx_edges(G, pos, alpha=0.5)
nx.draw_networkx_labels(G, pos, font_size=10)
plt.title("Louvain Community Detection on Karate Club Graph")
plt.axis("off")
plt.legend()
plt.show()
 
# 4. Print community assignments
print("Community assignments:")
for node, comm in partition.items():
    print(f"Node {node} → Community {comm}")