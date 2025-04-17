"""
Project 541: Relation Extraction with Distant Supervision
Description:
Relation extraction involves identifying and extracting relationships between entities in text. Distant supervision is a technique where a model is trained using labeled data obtained automatically from existing knowledge bases, even if that data is noisy or incomplete. In this project, we will extract relationships between entities in a given text using distant supervision.
"""

import spacy
from spacy import displacy
 
# 1. Load pre-trained SpaCy model for Named Entity Recognition (NER)
nlp = spacy.load("en_core_web_sm")
 
# 2. Provide a sample text (e.g., text with multiple entities and potential relations)
text = "Steve Jobs co-founded Apple in 1976. Bill Gates is the co-founder of Microsoft."
 
# 3. Process the text with SpaCy
doc = nlp(text)
 
# 4. Extract entities and potential relationships
entities = [(ent.text, ent.label_) for ent in doc.ents]
 
# 5. Simple rule-based extraction of relations using distance supervision
relations = []
for i in range(len(entities) - 1):
    entity1, entity2 = entities[i], entities[i+1]
    if entity1[1] == "PERSON" and entity2[1] == "ORG":
        relations.append((entity1[0], "co-founded", entity2[0]))
    elif entity1[1] == "PERSON" and entity2[1] == "PERSON":
        relations.append((entity1[0], "is associated with", entity2[0]))
 
# 6. Display the extracted relations
print("Extracted Relations:")
for relation in relations:
    print(f"{relation[0]} - {relation[1]} - {relation[2]}")