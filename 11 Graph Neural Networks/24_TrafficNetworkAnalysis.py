"""
Project 424. Traffic network analysis
Description:
Traffic network analysis models roads and intersections as graphs, where nodes are intersections and edges are roads. This enables analysis of traffic flow, congestion detection, routing, and urban planning. In this project, we’ll simulate a city traffic network using NetworkX and analyze key metrics such as shortest paths, bottlenecks, and edge centrality.

About:
✅ What It Does:
Models a simple city road map using a directed graph.

Computes shortest path between intersections based on road length.

Identifies bottleneck roads using edge betweenness centrality.

Visualizes the traffic network with labeled roads and distances.
"""

# pip install networkx matplotlib

import networkx as nx
import matplotlib.pyplot as plt
 
# 1. Create a directed weighted graph representing city roads
G = nx.DiGraph()
 
# Add intersections (nodes)
G.add_nodes_from(range(1, 10))
 
# Add roads (edges with traffic weights as 'length')
roads = [
    (1, 2, 5), (2, 3, 4), (3, 6, 8), (6, 9, 6),
    (1, 4, 6), (4, 5, 3), (5, 6, 4), (2, 5, 7),
    (4, 7, 5), (7, 8, 3), (8, 9, 2), (5, 8, 6)
]
G.add_weighted_edges_from(roads, weight='length')
 
# 2. Analyze shortest path from source to destination
source, destination = 1, 9
path = nx.shortest_path(G, source=source, target=destination, weight='length')
path_length = nx.shortest_path_length(G, source=source, target=destination, weight='length')
 
print(f"Shortest path from {source} to {destination}: {path}")
print(f"Total travel distance: {path_length} units")
 
# 3. Edge betweenness centrality (bottleneck roads)
centrality = nx.edge_betweenness_centrality(G, weight='length')
top_edges = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:3]
print("\nMost critical roads (by edge betweenness centrality):")
for edge, score in top_edges:
    print(f"Road {edge} → Centrality: {score:.4f}")
 
# 4. Visualize the network
pos = nx.spring_layout(G)
edge_labels = nx.get_edge_attributes(G, 'length')
nx.draw(G, pos, with_labels=True, node_color='lightgreen', node_size=600, arrows=True)
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
plt.title("Simulated City Traffic Network")
plt.show()