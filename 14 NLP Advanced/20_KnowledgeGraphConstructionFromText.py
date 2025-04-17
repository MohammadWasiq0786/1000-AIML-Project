"""
Project 540: Knowledge Graph Construction from Text
Description:
Knowledge graph construction involves extracting entities, relationships, and facts from unstructured text and representing them in a graph structure. This is useful for creating semantic networks, improving search engines, and enabling question answering systems. In this project, we will use NLP models to extract entities and relationships from text and construct a simple knowledge graph.
"""

import spacy
from spacy import displacy
import networkx as nx
import matplotlib.pyplot as plt
 
# 1. Load pre-trained SpaCy model for Named Entity Recognition (NER)
nlp = spacy.load("en_core_web_sm")
 
# 2. Provide a sample text (e.g., a sentence with entities and relationships)
text = "Albert Einstein was born in Ulm, Germany in 1879. He developed the theory of relativity."
 
# 3. Process the text with SpaCy
doc = nlp(text)
 
# 4. Extract entities (persons, places, dates, etc.)
entities = [(ent.text, ent.label_) for ent in doc.ents]
 
# 5. Build a simple knowledge graph (using NetworkX)
G = nx.Graph()
 
# Add nodes (entities)
for entity in entities:
    G.add_node(entity[0], label=entity[1])
 
# Add edges (relationships between entities based on sentence structure)
G.add_edge("Albert Einstein", "Ulm", relationship="born_in")
G.add_edge("Albert Einstein", "1879", relationship="born_in")
G.add_edge("Albert Einstein", "theory of relativity", relationship="developed")
 
# 6. Visualize the knowledge graph
plt.figure(figsize=(8, 8))
pos = nx.spring_layout(G, seed=42)
nx.draw(G, pos, with_labels=True, node_size=3000, node_color='lightblue', font_size=12, font_weight='bold')
plt.title("Knowledge Graph for Extracted Entities and Relationships")
plt.show()