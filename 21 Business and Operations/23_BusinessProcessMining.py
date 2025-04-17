"""
Project 823. Business Process Mining

Business process mining involves analyzing event logs to discover, monitor, and improve real business processes. It helps visualize how tasks are actually performed (vs. how they were designed), identify bottlenecks, and suggest automation. In this project, we simulate an event log and generate a simple process map using a directed graph.

This code visualizes the actual flow of business activities from the event log, revealing how often certain transitions occur. More advanced mining uses log timestamps, case durations, and conformance checking.
"""

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
 
# Simulated event log: each row = one step in a process
# CaseID identifies the process instance, Activity is the step taken
event_log = pd.DataFrame({
    'CaseID': [1, 1, 1, 2, 2, 3, 3, 3],
    'Activity': ['Start', 'Check Form', 'Approve', 'Start', 'Reject',
                 'Start', 'Check Form', 'Reject']
})
 
# Create edges based on activity transitions within each case
edges = []
for case_id in event_log['CaseID'].unique():
    activities = event_log[event_log['CaseID'] == case_id]['Activity'].tolist()
    edges += [(activities[i], activities[i+1]) for i in range(len(activities) - 1)]
 
# Count frequency of each transition
edge_freq = pd.Series(edges).value_counts().reset_index()
edge_freq.columns = ['Edge', 'Count']
 
# Create directed graph with frequencies
G = nx.DiGraph()
for edge, count in zip(edge_freq['Edge'], edge_freq['Count']):
    G.add_edge(edge[0], edge[1], weight=count, label=f'{count}x')
 
# Draw process map
pos = nx.spring_layout(G, seed=42)
plt.figure(figsize=(10, 6))
nx.draw(G, pos, with_labels=True, node_size=2000, node_color='lightblue', arrows=True)
edge_labels = nx.get_edge_attributes(G, 'label')
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
plt.title('Discovered Business Process Map')
plt.tight_layout()
plt.show()
 
# Show transitions
print("Activity Transitions:")
print(edge_freq)