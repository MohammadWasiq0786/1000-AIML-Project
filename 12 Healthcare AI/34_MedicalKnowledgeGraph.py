"""
Project 474. Medical knowledge graph
Description:
A Medical Knowledge Graph (KG) organizes relationships among medical entities like diseases, symptoms, drugs, and treatments. It enables reasoning, semantic search, and clinical decision support. In this project, we'll create a simple KG using NetworkX and simulate queries like "Which drugs treat hypertension?"

✅ What It Does:
Constructs a mini medical knowledge graph.

Supports queries like treatments, side effects, and connections.

Extendable to:

Parse real data from PubMed abstracts or clinical notes

Add reasoning with graph embeddings or LLMs

Deploy in clinical assistants for structured Q&A

Real-world sources:

UMLS, SNOMED CT, DrugBank, MeSH

Tools: Neo4j, RDF/SPARQL, KG-BERT for link prediction
"""

import networkx as nx
 
# 1. Create a directed graph
G = nx.DiGraph()
 
# 2. Add nodes (entities) and edges (relations)
G.add_edge("Hypertension", "Amlodipine", relation="treated_by")
G.add_edge("Hypertension", "Lifestyle Change", relation="managed_by")
G.add_edge("Diabetes", "Metformin", relation="treated_by")
G.add_edge("Metformin", "Nausea", relation="side_effect")
G.add_edge("Insulin", "Hypoglycemia", relation="side_effect")
G.add_edge("Obesity", "Lifestyle Change", relation="managed_by")
G.add_edge("Obesity", "Bariatric Surgery", relation="treated_by")
G.add_edge("Amlodipine", "Dizziness", relation="side_effect")
 
# 3. Query: What treats Hypertension?
def get_treatments(disease):
    return [tgt for src, tgt, rel in G.edges(data="relation") if src == disease and rel in ["treated_by", "managed_by"]]
 
print("Treatments for Hypertension:", get_treatments("Hypertension"))
 
# 4. Query: What are side effects of a drug?
def get_side_effects(drug):
    return [tgt for src, tgt, rel in G.edges(data="relation") if src == drug and rel == "side_effect"]
 
print("Side effects of Amlodipine:", get_side_effects("Amlodipine"))
 
# 5. Visualize graph (optional)
try:
    import matplotlib.pyplot as plt
    pos = nx.spring_layout(G)
    edge_labels = nx.get_edge_attributes(G, 'relation')
    nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=2500, font_size=10)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red')
    plt.title("Medical Knowledge Graph")
    plt.show()
except ImportError:
    print("Install matplotlib to visualize the graph.")