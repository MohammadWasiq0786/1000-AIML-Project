"""
Project 339. Graph-based recommendation system
Description:
A graph-based recommendation system represents users, items, and their relationships as a graph. In this graph:

Users and items are nodes

Interactions (e.g., ratings, clicks, purchases) are edges connecting users and items

This system can recommend items by:

Traversing the graph to find items connected to a user

Using graph algorithms like random walks or graph convolutional networks (GCNs) to model user-item relationships

In this project, we’ll implement a simple graph-based recommendation using a user-item interaction graph and a random walk to find similar items.

About:
✅ What It Does:
Creates a bipartite graph representing user-item interactions

Uses a random walk to explore the graph and recommend items to a user based on their connections

Recommends items that are frequently visited in the random walk for User1
"""

import numpy as np
import pandas as pd
import networkx as nx
 
# 1. Simulate user-item ratings matrix (user-item interactions as a graph)
users = ['User1', 'User2', 'User3', 'User4', 'User5']
items = ['Item1', 'Item2', 'Item3', 'Item4', 'Item5']
ratings = np.array([
    [5, 4, 0, 0, 3],
    [4, 0, 0, 3, 2],
    [1, 1, 0, 5, 4],
    [0, 0, 5, 4, 4],
    [2, 3, 0, 1, 0]
])
 
df = pd.DataFrame(ratings, index=users, columns=items)
 
# 2. Create a bipartite graph (users and items)
G = nx.Graph()
for user_idx, user in enumerate(users):
    for item_idx, item in enumerate(items):
        if df.iloc[user_idx, item_idx] > 0:  # If user has rated the item
            G.add_edge(user, items[item_idx], weight=df.iloc[user_idx, item_idx])  # Edge with rating as weight
 
# 3. Perform a simple random walk to recommend items for User1
def recommend_items_random_walk(user, G, top_n=3):
    # Start from the user node and perform a random walk
    neighbors = list(G.neighbors(user))  # Get items rated by the user
    recommended_items = []
 
    for _ in range(10):  # Perform 10 random walks
        current_node = user
        walk = []
        while current_node in G and len(walk) < 5:  # Limit walk length
            neighbors = list(G.neighbors(current_node))
            if not neighbors:
                break
            next_node = np.random.choice(neighbors)
            walk.append(next_node)
            current_node = next_node
        recommended_items.extend(walk)
 
    # Count item recommendations and sort by frequency
    recommended_items = pd.Series(recommended_items).value_counts().head(top_n).index.tolist()
    return recommended_items
 
# 4. Recommend items for User1 using random walk
user = 'User1'
recommended_items = recommend_items_random_walk(user, G)
print(f"Graph-Based Recommendations for {user}: {recommended_items}")