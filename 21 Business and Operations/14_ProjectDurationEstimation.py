"""
Project 814. Project Duration Estimation

Project duration estimation predicts how long a project will take based on task dependencies and time requirements. One widely used method is the Critical Path Method (CPM), which identifies the longest path through dependent tasks — this determines the shortest time in which the project can be completed.

This code builds a task dependency graph, computes the critical path, and estimates the total project duration. For real project planning, this can be expanded with resource allocation, buffer times, and probabilistic estimates (e.g., PERT).
"""

import networkx as nx
import matplotlib.pyplot as plt
 
# Define tasks with their durations and dependencies
tasks = {
    'A': {'duration': 4, 'deps': []},         # Start task
    'B': {'duration': 3, 'deps': ['A']},
    'C': {'duration': 2, 'deps': ['A']},
    'D': {'duration': 5, 'deps': ['B']},
    'E': {'duration': 2, 'deps': ['C']},
    'F': {'duration': 3, 'deps': ['D', 'E']}  # End task
}
 
# Create a directed graph to model task dependencies
G = nx.DiGraph()
 
# Add tasks and edges to the graph
for task, info in tasks.items():
    G.add_node(task, duration=info['duration'])
    for dep in info['deps']:
        G.add_edge(dep, task)
 
# Compute the longest path (critical path)
critical_path = nx.dag_longest_path(G, weight='duration')
critical_duration = nx.dag_longest_path_length(G, weight='duration')
 
# Display results
print("Critical Path:", ' -> '.join(critical_path))
print(f"Estimated Project Duration: {critical_duration} days")
 
# Draw the graph
pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=1500)
labels = {node: f"{node}\n({G.nodes[node]['duration']}d)" for node in G.nodes}
nx.draw_networkx_labels(G, pos, labels=labels)
plt.title("Project Task Dependency Graph")
plt.tight_layout()
plt.show()